import io
import grpc
import comm_pb2
import comm_pb2_grpc
import sys
import torch

sys.path.append('../src/')

from utils.model_utils import ModelUtils
from grpc_utils import deserialize_model, serialize_model

host = 'matlaber7.media.mit.edu:50051'

def run_client():
    # Compute local average
    model = ModelUtils().get_model("resnet18", "cifar10", "cuda:2", [2])

    # Serialize the model state dictionary
    model_bytes = serialize_model(model)

    print('Model size:', len(model_bytes))
    # Connect to server and send local average
    with grpc.insecure_channel(host) as channel:
        stub = comm_pb2_grpc.CommunicationServerStub(channel)
        user_id = stub.GetID(comm_pb2.Empty())
        print('user got', user_id.id, user_id.num)
        response = stub.SendMessage(comm_pb2.Message(model=comm_pb2.Model(buffer=model_bytes)))
        print('Submitted model')
        g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
        print('Got model')

if __name__ == '__main__':
    run_client()
