import gc
from concurrent import futures
import io
import threading
from typing import List
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
import grpc
import numpy as np
import comm_pb2
import comm_pb2_grpc
import torch
import torch.nn as nn
import sys

sys.path.append('../src/')
from grpc_utils import deserialize_model, serialize_model
from utils.model_utils import ModelUtils

class Constants:
    def __init__(self):
        self.STATUS_ALIVE = 'ALIVE'
        self.STATUS_DEAD = 'DEAD'

const = Constants()

class CommunicationServicer(comm_pb2_grpc.CommunicationServer):
    def __init__(self):
        self.local_averages: List[OrderedDictType] = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.target_clients = 4
        self.model = ModelUtils().get_model("resnet18", "cifar10", "cuda:1", [1])
        self.user_ids = {}

    def _generate_ID(self):
        """ Generate a random hex ID for a user
        Needs to be thread-safe, although it's unlikely
        collision will occur due to the large range of IDs
        """
        int_id = np.random.randint(0, 2**31 - 1)  # Generate a random integer between 0 and 2^31 - 1
        random_id = hex(hash(int_id))
        while random_id in self.user_ids:
            int_id = np.random.randint(0, 2**31 - 1)
            random_id = hex(hash(int_id))
        return random_id

    def GetID(self, request, context):
        # whenever a user connects, assign them an random hex ID
        with self.lock:
            random_id = self._generate_ID()
            user_num = len(self.user_ids) + 1
            self.user_ids[random_id] = {'user_num': user_num, 'status': const.STATUS_ALIVE}
            print(f'User {user_num} connected with ID {random_id}')
        return comm_pb2.ID(id=random_id, num=user_num)

    def GetSize(self, request, context):
        return comm_pb2.Size(size=len(self.user_ids.keys()))

    def GetModel(self, request, context):
        return comm_pb2.Model(model=serialize_model(self.model))

    def SendMessage(self, request, context):
        with self.lock:
            self.local_averages.append(deserialize_model(request.model.buffer))
            self.num_clients = len(self.local_averages)
            if self.num_clients >= self.target_clients:
                self.condition.notify_all()

        with self.condition:
            while self.num_clients < self.target_clients:
                self.condition.wait()

        self._compute_global_average()
        self.local_averages: List[OrderedDictType] = []

        # free up memory
        gc.collect()
        torch.cuda.empty_cache()

        return comm_pb2.Empty()

    def _compute_global_average(self):
        if not self.local_averages:
            return None

        avg_state_dict = OrderedDict()

        # Sum up the parameters from each local model's state dictionary
        for state_dict in self.local_averages:
            for name, param in state_dict.items():
                if name not in avg_state_dict:
                    avg_state_dict[name] = param.clone()
                else:
                    avg_state_dict[name] += param
        # Divide the summed parameters by the number of local models to get the average
        for name in avg_state_dict:
            try:
                avg_state_dict[name] /= float(len(self.local_averages))
            except:
                # BUG: This happens for batchnorm num_batches_tracked parameter
                # which is a long tensor and cannot be divided by a float
                # So we just skip it
                # TODO: Adjust the client code to not send this parameter
                pass
        # Load the averaged state dictionary into the global model
        self.model.load_state_dict(avg_state_dict)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
    ])
    comm_pb2_grpc.add_CommunicationServerServicer_to_server(CommunicationServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Server started')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
