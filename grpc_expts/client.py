import os
import grpc
import comm_pb2
import comm_pb2_grpc
import sys
import torch
import numpy as np

BASE_DIR = '/u/abhi24/Workspace/SONAR'
sys.path.append(f'{BASE_DIR}/src/')

from utils.data_utils import get_dataset
from utils.model_utils import ModelUtils
from utils.log_utils import LogUtils
from torch.utils.data import DataLoader, Subset
from grpc_utils import deserialize_model, serialize_model # type: ignore

host = 'matlaber7.media.mit.edu:50051'

TEMP_TOTAL_NODES = 4
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
LOG_BASE_DIR = f'{BASE_DIR}/grpc_expts/logs'
config = {}

def run_client():
    # Compute local average
    device_offset = 1
    dset = 'cifar10'
    dpath = f'{BASE_DIR}/data'
    model_utils = ModelUtils()
    loss_fn = torch.nn.CrossEntropyLoss()

    # Connect to server and send local average
    with grpc.insecure_channel(host, options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024), # 50MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024) # 50MB
    ]) as channel:
        stub = comm_pb2_grpc.CommunicationServerStub(channel)
        user_id = stub.GetID(comm_pb2.Empty())
        config['log_path'] = f'{LOG_BASE_DIR}/{dset}/user_{user_id.num}'
        # try to create a new log directory, if it already exists then we overwrite!!
        # TODO: shouldn't be overwriting
        try:
            os.mkdir(config['log_path'])
        except FileExistsError:
            pass
        config['load_existing'] = False
        log_utils = LogUtils(config)
        log_utils.log_console(f'user got {user_id.id} {user_id.num}')
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
        te_loss, te_acc = model_utils.test(model, test_loader, loss_fn, device)
        log_utils.log_console(f'Test Loss: {te_loss:.3f}, Test Acc: {te_acc:.3f}')
        log_utils.log_tb('test_loss', te_loss, 0)
        log_utils.log_tb('test_acc', te_acc, 0)
        optim = torch.optim.Adam(model.parameters(), lr=3e-4)

        g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
        g_model = deserialize_model(g_model.buffer)
        model.load_state_dict(g_model, strict=False)
        log_utils.log_console('received model from server')

        for i in range(1, 101):
            tr_loss, tr_acc = model_utils.train(model, optim, dloader, loss_fn, device)
            log_utils.log_console(f'Epoch {i}, Node {user_id.num}, Loss: {tr_loss:.3f}, Acc: {tr_acc:.3f}')

            # Send model to server
            model_bytes = serialize_model(model.state_dict())
            stub.SendMessage(comm_pb2.Message(
                model=comm_pb2.Model(buffer=model_bytes),
                id=user_id.id
            ))

            # Get global model from server
            g_model = stub.GetModel(comm_pb2.ID(id=user_id.id, num=user_id.num))
            g_model = deserialize_model(g_model.buffer)

            # Load global model
            model.load_state_dict(g_model, strict=False)

            # Test global model
            te_loss, te_acc = model_utils.test(model, test_loader, loss_fn, device)
            log_utils.log_console(f'Test Loss: {te_loss:.3f}, Test Acc: {te_acc:.3f}')
            log_utils.log_tb('train_loss', tr_loss, i)
            log_utils.log_tb('train_acc', tr_acc, i)
            log_utils.log_tb('test_loss', te_loss, i)
            log_utils.log_tb('test_acc', te_acc, i)

        stub.SendBye(comm_pb2.ID(id=user_id.id))
        print('Exiting...', user_id.num, f'{te_acc:.3f}')

if __name__ == '__main__':
    run_client()
