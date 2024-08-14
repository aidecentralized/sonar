from concurrent import futures
from queue import Queue
from typing import Any, Dict, List, OrderedDict
import grpc # type: ignore
from utils.communication.grpc.grpc_utils import deserialize_model, serialize_model
import os
import sys

grpc_generated_dir = os.path.dirname(os.path.abspath(__file__))
if grpc_generated_dir not in sys.path:
    sys.path.append(grpc_generated_dir)

import comm_pb2 as comm_pb2
import comm_pb2_grpc as comm_pb2_grpc
from utils.communication.interface import CommunicationInterface

class Servicer(comm_pb2_grpc.CommunicationServerServicer):
    def __init__(self):
        self.received_data: Queue[Any] = Queue()

    def SendData(self, request, context) -> comm_pb2.Empty: # type: ignore
        self.received_data.put(deserialize_model(request.model.buffer)) # type: ignore
        return comm_pb2.Empty() # type: ignore

class GRPCCommunication(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        # TODO: Implement this differently later by creating a super node
        # that maintains a list of all peers
        # all peers will send their IDs to this super node
        # when they start.
        self.all_peer_ids: List[str] = config["comm"]["all_peer_ids"]
        self.rank: int = config["comm"]["rank"]
        address: str = str(self.all_peer_ids[self.rank])
        self.port: int = int(address.split(":")[1])
        self.host: str = address.split(":")[0]
        self.server = None
        self.servicer = Servicer()

        # remove self from the list of all peer IDs
        self.all_peer_ids.remove(self.host + ":" + str(self.port))

    def initialize(self):
        self.listener: Any = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[ # type: ignore
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ])
        comm_pb2_grpc.add_CommunicationServerServicer_to_server(self.servicer, self.listener) # type: ignore
        self.listener.add_insecure_port(f'[::]:{self.port}')
        self.listener.start()
        print(f'Started server on port {self.port}')

    def send(self, dest: str|int, data: OrderedDict[str, Any]):
        """
        data should be a torch model
        """
        dest_host: str = ""
        if type(dest) == int:
            dest_host = self.all_peer_ids[int(dest)]
        else:
            dest_host = str(dest)
        try:
            buffer = serialize_model(data)
            with grpc.insecure_channel(dest_host) as channel: # type: ignore
                stub = comm_pb2_grpc.CommunicationServerStub(channel) # type: ignore
                model = comm_pb2.Model(buffer=buffer) # type: ignore
                stub.SendData(comm_pb2.Data(model=model, id="tempID")) # type: ignore
        except grpc.RpcError as e:
            print(f"RPC failed: {e}")
            sys.exit(1)

    def receive(self, node_ids: str|int) -> Any:
        # this .get() will block until
        # at least 1 item is received in the queue
        return self.servicer.received_data.get()

    def broadcast(self, data: Any):
        for peer_id in self.all_peer_ids:
            self.send(peer_id, data)

    def all_gather(self) -> Any:
        # this will block until all items are received
        # from all peers
        items: List[Any] = []
        for peer_id in self.all_peer_ids:
            items.append(self.receive(peer_id))
        return items

    def finalize(self):
        if self.listener:
            self.listener.stop(0)
            print(f'Stopped server on port {self.port}')
