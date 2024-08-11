from mpi4py import MPI
from utils.communication.comm_utils import CommunicationInterface

class MPICommUtils(CommunicationInterface):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def initialize(self):
        pass

    def send(self, dest, data):
        self.comm.send(data, dest=dest)
            
    def receive(self, node_ids, data):
        return self.comm.recv(source=node_ids)

    def broadcast(self, data):
        for i in range(1, self.size):
            if i != self.rank:
                self.send(i, data)

    def finalize(self):
        pass        
