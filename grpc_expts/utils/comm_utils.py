import grpc
import comm_pb2
import comm_pb2_grpc

class CommUtils:
    def __init__(self, server_address):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = comm_pb2_grpc.CommunicationServerStub(self.channel)
        self.client_id = self.stub.GetID(comm_pb2.Empty()).id

    def send_signal(self, data):
        message = comm_pb2.Message(id=self.client_id, model=comm_pb2.Model(buffer=data))
        self.stub.SendMessage(message)
    
    def send_signal_to_all_clients(self, client_ids, data):
        for client_id in client_ids:
            message = comm_pb2.Message(id=client_id, model=comm_pb2.Model(buffer=data))
            self.stub.SendMessage(message)

    def wait_for_signal(self):
        response = self.stub.GetModel(comm_pb2.ID(id=self.client_id))
        return response.buffer

    def wait_for_all_clients(self, client_ids):
        data_list = []
        for client_id in client_ids:
            response = self.stub.GetModel(comm_pb2.ID(id=client_id))
            data_list.append(response.buffer)
        return data_list
