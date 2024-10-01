from typing import Dict, Any, List
from mpi4py import MPI
from utils.communication.interface import CommunicationInterface


class MPICommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def initialize(self):
        pass

    def send(self, dest: str | int, data: Any):
        self.comm.send(data, dest=int(dest))

    def receive(self, node_ids: str | int) -> Any:
        return self.comm.recv(source=int(node_ids))

    def broadcast(self, data: Any):
        for i in range(1, self.size):
            if i != self.rank:
                self.send(i, data)

    def all_gather(self):
        """
        This function is used to gather data from all the nodes.
        """
        items: List[Any] = []
        for i in range(1, self.size):
            items.append(self.receive(i))
        return items

    def finalize(self):
        pass
