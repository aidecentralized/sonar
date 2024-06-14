import grpc
import comm_pb2
import comm_pb2_grpc
import sys
import torch
import numpy as np

sys.path.append('../src/')

from utils.data_utils import get_dataset
from utils.model_utils import ModelUtils
from torch.utils.data import DataLoader, Subset
from grpc_utils import deserialize_model, serialize_model # type: ignore

host = 'matlaber7.media.mit.edu:50051'

TEMP_TOTAL_NODES = 4
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

def run_client():
    # Compute local average
    device_offset = 1
    dset = 'cifar10'
    dpath = '../data'
    model_utils = ModelUtils()
    loss_fn = torch.nn.CrossEntropyLoss()

    # Connect to server and send local average
    with grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024), # 50MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024) # 50MB
    ]) as channel:
        stub = comm_pb2_grpc.CommunicationServerStub(channel)
        user_id = stub.GetID(comm_pb2.Empty())
        print('user got', user_id.id, user_id.num)
        node_id = user_id.num % TEMP_TOTAL_NODES
        device = f'cuda:{node_id + device_offset}'
        dset_obj = get_dataset(dset, dpath=dpath)
        train_dset = dset_obj.train_dset
        indices = np.random.permutation(len(train_dset))
        samples_per_client = 1000
        train_indices = indices[node_id*samples_per_client:(node_id+1)*samples_per_client]
        # print('train_indices', train_indices, f'Node {user_id.num}')
        train_dset = Subset(train_dset, train_indices)
        dloader = DataLoader(train_dset, batch_size=64, shuffle=True)
        test_dset = dset_obj.test_dset
        test_loader = DataLoader(test_dset, batch_size=64)

        model = model_utils.get_model("resnet18", "cifar10", device, [])
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        for i in range(100):
            tr_loss, tr_acc = model_utils.train(model, optim, dloader, loss_fn, device)
            print(f'Epoch {i}, Node {user_id.num}, Loss: {tr_loss}, Acc: {tr_acc}')
            model_bytes = serialize_model(model)
            stub.SendMessage(comm_pb2.Message(model=comm_pb2.Model(buffer=model_bytes)))
            g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
            g_model = deserialize_model(g_model.buffer)
            model.load_state_dict(g_model)
            te_loss, te_acc = model_utils.test(model, test_loader, loss_fn, device)
            print(f'Test Loss: {te_loss}, Test Acc: {te_acc}')
        print('Got model')

if __name__ == '__main__':
    run_client()
