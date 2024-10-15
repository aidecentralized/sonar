from __future__ import annotations
from concurrent import futures
from queue import Queue
import random
import re
import threading
import time
import socket
from typing import Any, Callable, Dict, List, OrderedDict, Union, TYPE_CHECKING
from urllib.parse import unquote
import grpc # type: ignore
from torch import Tensor
from utils.communication.grpc.grpc_utils import deserialize_model, serialize_model
import os
import sys

grpc_generated_dir = os.path.dirname(os.path.abspath(__file__))
if grpc_generated_dir not in sys.path:
    sys.path.append(grpc_generated_dir)

import comm_pb2 as comm_pb2  # noqa: E402
import comm_pb2_grpc as comm_pb2_grpc  # noqa: E402
from utils.communication.interface import CommunicationInterface  # noqa: E402

if TYPE_CHECKING:
    from algos.base_class import BaseNode


MAX_MESSAGE_LENGTH = 200 * 1024 * 1024 # 200MB

# TODO: Several changes needed to improve the quality of the code
# 1. We need to improve comm.proto and get rid of singletons like Rank, Port etc.
# 2. Some parts of the code are heavily nested and need to be refactored
# 3. Insert try-except blocks wherever communication is involved
# 4. Probably a good idea to move the Servicer class to a separate file
# 5. Not needed for benchmarking but for the system to be robust, we need to implement timeouts and fault tolerance
# 6. Peer_ids should be indexed by a unique identifier
# 7. Try to get rid of type: ignore as much as possible


def is_port_available(port: int) -> bool:
    """
    Check if a port is available for use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # type: ignore
        return s.connect_ex(("localhost", port)) != 0  # type: ignore


def get_port(rank: int, num_users: int) -> int:
    """
    Get the port number for the given rank.
    """
    start = 50051
    while True:
        port = start + rank
        if is_port_available(port):
            return port
        # if we increment by 1 then it's likely that
        # the next node will also have the same port number
        start += num_users
        if start > 65535:
            raise Exception(f"No available ports for node {rank}")


def parse_peer_address(peer: str) -> str:
    # Remove 'ipv4:' or 'ipv6:' prefix
    if peer.startswith(("ipv4:", "ipv6:")):
        peer = peer.split(":", 1)[1]

    # Handle IPv6 address
    if peer.startswith("["):
        # Extract IPv6 address
        match = re.match(r"\[([^\]]+)\]", peer)
        if match:
            return unquote(match.group(1))  # Decode URL-encoded characters
    else:
        # Handle IPv4 address or hostname
        return peer.rsplit(":", 1)[0]  # Remove port number
    return ""


class Servicer(comm_pb2_grpc.CommunicationServerServicer):
    def __init__(self, super_node_host: str):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.received_data: Queue[Any] = Queue()
        self.quorum: Queue[bool] = Queue()
        self.finished: Queue[int] = Queue()
        port = int(super_node_host.split(":")[1])
        ip = super_node_host.split(":")[0]
        self.base_node: BaseNode | None = None
        self.peer_ids: OrderedDict[int, Dict[str, int | str]] = OrderedDict(
            {0: {"rank": 0, "port": port, "ip": ip}}
        )

    def register_self(self, obj: "BaseNode"):
        self.base_node = obj

    def send_data(self, request, context) -> comm_pb2.Empty:  # type: ignore
        self.received_data.put(deserialize_model(request.model.buffer))  # type: ignore
        return comm_pb2.Empty()  # type: ignore

    def get_rank(self, request: comm_pb2.Empty, context: grpc.ServicerContext) -> comm_pb2.Rank | None:  
        try:
            with self.lock:
                peer = context.peer()  # type: ignore
                # parse the hostname from peer
                peer_str = parse_peer_address(peer)  # type: ignore
                rank = len(self.peer_ids)
                # TODO: index the peer_ids by a unique identifier
                self.peer_ids[rank] = {"rank": rank, "port": 0, "ip": peer_str}
                rank = self.peer_ids[rank].get("rank", -1)  # Default to -1 if not found
            return comm_pb2.Rank(rank=rank)  # type: ignore
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Error in get_rank: {str(e)}")  # type: ignore

    def get_model(self, request: comm_pb2.Empty, context: grpc.ServicerContext) -> comm_pb2.Model | None:
        if not self.base_node:
            context.abort(grpc.StatusCode.INTERNAL, "Base node not registered") # type: ignore
            raise Exception("Base node not registered")
        with self.lock:
            model = comm_pb2.Model(buffer=serialize_model(self.base_node.get_model_weights()))
            return model

    def get_current_round(self, request: comm_pb2.Empty, context: grpc.ServicerContext) -> comm_pb2.Round | None:
        if not self.base_node:
            context.abort(grpc.StatusCode.INTERNAL, "Base node not registered") # type: ignore
            raise Exception("Base node not registered")
        with self.lock:
            round = comm_pb2.Round(round=self.base_node.get_local_rounds())
            return round

    def update_port(
        self, request: comm_pb2.PeerIds, context: grpc.ServicerContext
    ) -> comm_pb2.Empty:
        with self.lock:
            # FIXME: This is a security vulnerability because
            # any node can update the ip and port of any other node
            self.peer_ids[request.rank.rank]["ip"] = request.ip  # type: ignore
            self.peer_ids[request.rank.rank]["port"] = request.port.port  # type: ignore
            return comm_pb2.Empty()  # type: ignore

    def send_peer_ids(self, request: comm_pb2.PeerIds, context) -> comm_pb2.Empty:  # type: ignore
        """
        Used by the super node to update all peers with the peer_ids
        after achieving quorum.
        """
        peer_ids: comm_pb2.PeerIds = request.peer_ids  # type: ignore
        for rank in peer_ids:  # type: ignore
            peer_id_proto = peer_ids[rank]  # type: ignore
            peer_id_dict: Dict[str, Union[int, str]] = {
                "rank": peer_id_proto.rank.rank,  # type: ignore
                "port": peer_id_proto.port.port,  # type: ignore
                "ip": peer_id_proto.ip,  # type: ignore
            }
            self.peer_ids[rank] = peer_id_dict
        return comm_pb2.Empty()

    def send_quorum(self, request, context) -> comm_pb2.Empty:  # type: ignore
        self.quorum.put(request.quorum)  # type: ignore
        return comm_pb2.Empty()  # type: ignore

    def send_finished(self, request, context) -> comm_pb2.Empty: # type: ignore
        self.finished.put(request.rank)  # type: ignore
        return comm_pb2.Empty()  # type: ignore

class GRPCCommunication(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        # TODO: Implement this differently later by creating a super node
        # that maintains a list of all peers
        # all peers will send their IDs to this super node
        # when they start.
        # The implementation will have
        # 1. Registration phase where every peer registers itself and gets a rank
        # 2. Once a threshold number of peers have registered, the super node sets quorum to True
        # 3. The super node broadcasts the peer_ids to all peers
        # 4. The nodes will execute rest of the protocol in the same way as before
        self.num_users: int = int(config["num_users"])  # type: ignore
        self.rank: int | None = config["comm"]["rank"]
        # TODO: Get rid of peer_ids now that we are passing [comm][host]
        self.super_node_host: str = config["comm"]["peer_ids"][0]
        self.synchronous: bool = config["comm"].get("synchronous", True)
        if self.rank == 0:
            node_id: List[str] = self.super_node_host.split(":")
            self.host: str = node_id[0]
            self.port: int = int(node_id[1])
        else:
            # get hostname based on ip address
            self.host: str = config["comm"]["host"]
            pass
        self.servicer = Servicer(self.super_node_host)

    @staticmethod
    def get_registered_users(peer_ids: OrderedDict[int, Dict[str, int | str]]) -> int:
        # count the number of entries that have a non-zero port
        return len(
            [peer_id for peer_id, values in peer_ids.items() if values.get("port") != 0]
        )

    def register_self(self, obj: "BaseNode"):
        self.servicer.register_self(obj)

    def recv_with_retries(self, host: str, callback: Callable[[comm_pb2_grpc.CommunicationServerStub], Any]) -> Any:
        with grpc.insecure_channel(host, options=[ # type: ignore
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:
            stub = comm_pb2_grpc.CommunicationServerStub(channel)
            max_tries = 10
            while max_tries > 0:
                try:
                    result = callback(stub)
                except grpc.RpcError as e:
                    print(f"RPC failed {10 - max_tries} times: {e}", "Retrying...")
                    # sleep for a random time between 1 and 10 seconds
                    random_time = random.randint(1, 10)
                    time.sleep(random_time)
                    max_tries -= 1
                else:
                    return result

    def register(self):
        """
        The registration protocol is as follows:
        1. Every node sends a request to the super node to get a rank
        2. The super node assigns a rank to the node and sends it back
        3. The node updates its port and sends the updated peer_ids to the super node
        """
        def callback_fn(stub: comm_pb2_grpc.CommunicationServerStub) -> int:
            rank_data = stub.get_rank(comm_pb2.Empty()) # type: ignore
            return rank_data.rank # type: ignore

        self.rank = self.recv_with_retries(self.super_node_host, callback_fn)
        self.port = get_port(self.rank, self.num_users)  # type: ignore because we are setting it in the register method
        rank = comm_pb2.Rank(rank=self.rank)  # type: ignore
        port = comm_pb2.Port(port=self.port)
        peer_id = comm_pb2.PeerId(rank=rank, port=port, ip=self.host)

        with grpc.insecure_channel(self.super_node_host) as channel:  # type: ignore
            stub = comm_pb2_grpc.CommunicationServerStub(channel)
            stub.update_port(peer_id)  # type: ignore

    def start_listener(self):
        self.listener: grpc.Server = grpc.server(  # type: ignore
            futures.ThreadPoolExecutor(max_workers=4),
            options=[  # type: ignore
                ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
            ],
        )
        comm_pb2_grpc.add_CommunicationServerServicer_to_server(self.servicer, self.listener)  # type: ignore
        address = f"{self.host}:{self.port}"
        self.listener.add_insecure_port(address)
        self.listener.start()
        print(f"Started listener on {address}")

    def peer_ids_to_proto(
        self, peer_ids: OrderedDict[int, Dict[str, int | str]]
    ) -> Dict[int, comm_pb2.PeerId]:
        peer_ids_proto: Dict[int, comm_pb2.PeerId] = {}
        for peer_id in peer_ids:
            rank = comm_pb2.Rank(rank=peer_ids[peer_id].get("rank"))  # type: ignore
            port = comm_pb2.Port(port=peer_ids[peer_id].get("port"))  # type: ignore
            ip = str(peer_ids[peer_id].get("ip"))
            peer_ids_proto[peer_id] = comm_pb2.PeerId(rank=rank, port=port, ip=ip)
        return peer_ids_proto

    def initialize(self):
        if self.rank != 0:
            self.register()

        self.start_listener()

        # wait for the quorum to be set
        if self.rank != 0:
            print(f"{self.rank} Waiting for quorum to be set")
            status = self.servicer.quorum.get()
            if not status:
                print("Quorum became false!")
                sys.exit(1)
        else:
            quorum_threshold = self.num_users + 1  # +1 for the super node
            num_registered = self.get_registered_users(self.servicer.peer_ids)
            while num_registered < quorum_threshold:
                # sleep for 5 seconds
                print(
                    f"Waiting for {quorum_threshold} users to register, {num_registered} have registered so far"
                )
                time.sleep(5)
                num_registered = self.get_registered_users(self.servicer.peer_ids)
                # TODO: Implement a timeout here and if the timeout is reached
                # then set quorum to False for all registered users
                # and exit the program

            print("All users have registered", self.servicer.peer_ids)
            for peer_id in self.servicer.peer_ids:
                host_ip = self.servicer.peer_ids[peer_id].get("ip")
                if peer_id != self.rank:
                    port = self.servicer.peer_ids[peer_id].get("port")
                    address = f"{host_ip}:{port}"
                    print(f"Sending peer_ids to {address}")
                    with grpc.insecure_channel(address) as channel:  # type: ignore
                        stub = comm_pb2_grpc.CommunicationServerStub(channel)
                        proto_msg = comm_pb2.PeerIds(
                            peer_ids=self.peer_ids_to_proto(self.servicer.peer_ids)
                        )
                        stub.send_peer_ids(proto_msg)  # type: ignore
                        stub.send_quorum(comm_pb2.Quorum(quorum=True))  # type: ignore

    def get_host_from_rank(self, rank: int) -> str:
        for peer_id in self.servicer.peer_ids:
            if self.servicer.peer_ids[peer_id].get("rank") == rank:
                return self.servicer.peer_ids[peer_id].get("ip") + ":" + str(self.servicer.peer_ids[peer_id].get("port"))  # type: ignore
        raise Exception(f"Rank {rank} not found in peer_ids")

    def send(self, dest: str | int, data: OrderedDict[str, Tensor]):
        """
        data should be a torch model
        """
        dest_host: str = ""
        if type(dest) is int:
            dest_host = self.get_host_from_rank(dest)
        else:
            dest_host = str(dest)
        try:
            buffer = serialize_model(data)
            with grpc.insecure_channel(dest_host) as channel:  # type: ignore
                stub = comm_pb2_grpc.CommunicationServerStub(channel)  # type: ignore
                model = comm_pb2.Model(buffer=buffer)  # type: ignore
                stub.send_data(comm_pb2.Data(model=model, id=str(self.rank)))  # type: ignore
        except grpc.RpcError as e:
            print(f"RPC failed: {e}")
            sys.exit(1)

    def wait_until_rounds_match(self, id: int):
        """
        Wait until the rounds match with the given id
        """
        if not self.servicer.base_node:
            raise Exception("Base node not registered")
        self_round = self.servicer.base_node.get_local_rounds()
        def callback_fn(stub: comm_pb2_grpc.CommunicationServerStub) -> int:
            round = stub.get_current_round(comm_pb2.Empty()) # type: ignore
            return round.round # type: ignore
        
        while True:
            host = self.get_host_from_rank(id)
            round = self.recv_with_retries(host, callback_fn)
            if round >= self_round:
                # Strict equality can not be enforced because
                # the communication is not symmetric
                # For example, if node 1 goes ahead with round 3
                # after getting weights from node 2 and then node 3
                # tries to get weights from node 1, it will be stuck
                # this will be only possible if everyone waits for their
                # 'subscribers' to catch up.
                break
            self.servicer.base_node.log_utils.log_console(
                f"Node {self.rank} Waiting for round {self_round} to match with Node {id}"
            )
            time.sleep(2)

    # TODO: We are using Any because we want to support any type of data
    # However, it is sensible to restrict the type and make different functions
    # for different types of data because we probably only have three categories
    # 1. Model weights - Dictionary of tensors
    # 2. Tensor data - Tensors
    # 3. Metadata - JSON format
    def receive(self, node_ids: List[int]) -> List[Any]:
        if self.synchronous:
            for id in node_ids:
                self.wait_until_rounds_match(id)
        items: List[Any] = []
        def callback_fn(stub: comm_pb2_grpc.CommunicationServerStub) -> OrderedDict[str, Tensor]:
            model = stub.get_model(comm_pb2.Empty()) # type: ignore
            return deserialize_model(model.buffer) # type: ignore

        for id in node_ids:
            rank = self.get_host_from_rank(id)
            item = self.recv_with_retries(rank, callback_fn)
            if item is None:
                raise Exception(f"Received None from node {id} by node {self.rank}", "Exiting...")
            items.append(item)
        return items

    def is_own_id(self, peer_id: int) -> bool:
        rank = self.servicer.peer_ids[peer_id].get("rank")
        if rank != self.rank:
            return False
        return True

    def broadcast(self, data: Any):
        for peer_id in self.servicer.peer_ids:
            if not self.is_own_id(peer_id):
                self.send(peer_id, data)

    def all_gather(self) -> Any:
        # this will block until all items are received
        # from all peers
        items: List[Any] = []
        for peer_id in self.servicer.peer_ids:
            if not self.is_own_id(peer_id):
                items.append(self.receive([peer_id])[0])
        return items

    def get_num_finished(self) -> int:
        num_finished = self.servicer.finished.qsize()
        return num_finished

    def finalize(self):
        # 1. All nodes send finished to the super node
        # 2. super node will wait for all nodes to send finished
        # 3. super node will then send bye to all nodes
        # 4. all nodes will wait for the bye and then exit
        # this is to ensure that all nodes have finished
        # and no one leaves early
        if self.rank == 0:
            quorum_threshold = self.num_users  # No +1 for the super node because it doesn't send finished
            num_finished = self.get_num_finished()
            while num_finished < quorum_threshold:
                # sleep for 5 seconds
                print(
                    f"Waiting for {quorum_threshold} users to finish, {num_finished} have finished so far"
                )
                time.sleep(5)
                num_finished = self.get_num_finished()

            # send quorum to all nodes
            for peer_id in self.servicer.peer_ids:
                if not self.is_own_id(peer_id):
                    host = self.get_host_from_rank(peer_id)
                    with grpc.insecure_channel(host) as channel: # type: ignore
                        stub = comm_pb2_grpc.CommunicationServerStub(channel)
                        stub.send_quorum(comm_pb2.Quorum(quorum=True)) # type: ignore
        else:
            # send finished to the super node
            with grpc.insecure_channel(self.super_node_host) as channel: # type: ignore
                stub = comm_pb2_grpc.CommunicationServerStub(channel)
                stub.send_finished(comm_pb2.Rank(rank=self.rank)) # type: ignore
            status = self.servicer.quorum.get()
            if not status:
                print("Quorum became false!")
                sys.exit(1)
        if self.listener:
            self.listener.stop(0)  # type: ignore
            print(f"Stopped server on port {self.port}")
