from concurrent import futures
import queue
from typing import Any, Dict
import grpc
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
        self.received_data = queue.Queue()

    def SendData(self, request, context):
        self.received_data.put(deserialize_model(request.model.buffer))
        return comm_pb2.Empty()

class GRPCCommunication(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.port: int = int(config["comm"]["port"])
        self.server = None
        self.servicer = Servicer()
        # TODO: Implement this later by creating a super node
        # that maintains a list of all peers
        # all peers will send their IDs to this super node
        # when they start.
        self.all_peer_ids = []

    def initialize(self):
        self.listener = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        comm_pb2_grpc.add_CommunicationServerServicer_to_server(self.servicer, self.listener)
        self.listener.add_insecure_port(f'[::]:{self.port}')
        self.listener.start()
        print(f'Started server on port {self.port}')

    def send(self, dest, data):
        """
        data should be a torch model
        """
        try:
            buffer = serialize_model(data.state_dict())
            with grpc.insecure_channel(dest) as channel:
                stub = comm_pb2_grpc.CommunicationServerStub(channel)
                stub.SendData(comm_pb2.Model(buffer=buffer))
        except grpc.RpcError as e:
            print(f"RPC failed: {e}")

    def receive(self, node_ids):
        # this .get() will block until
        # at least 1 item is received in the queue
        return self.servicer.received_data.get()

    def broadcast(self, data):
        for peer_id in self.all_peer_ids:
            self.send(peer_id, data)

    def finalize(self):
        if self.listener:
            self.listener.stop(0)
            print(f'Stopped server on port {self.port}')
