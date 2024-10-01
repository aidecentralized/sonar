from enum import Enum
from typing import Any, Dict, List

from utils.communication.grpc.main import GRPCCommunication
from utils.communication.mpi import MPICommUtils


class CommunicationType(Enum):
    MPI = 1
    GRPC = 2
    HTTP = 3


class CommunicationFactory:
    @staticmethod
    def create_communication(
        config: Dict[str, Any], comm_type: CommunicationType
    ) -> Any:
        comm_type = comm_type
        if comm_type == CommunicationType.MPI:
            return MPICommUtils(config)
        elif comm_type == CommunicationType.GRPC:
            return GRPCCommunication(config)
        elif comm_type == CommunicationType.HTTP:
            raise NotImplementedError("HTTP communication not yet implemented")
        else:
            raise ValueError("Invalid communication type")


class CommunicationManager:
    def __init__(self, config: Dict[str, Any]):
        self.comm_type = CommunicationType[config["comm"]["type"]]
        self.comm = CommunicationFactory.create_communication(config, self.comm_type)
        self.comm.initialize()

    def get_rank(self):
        if self.comm_type == CommunicationType.MPI:
            return self.comm.rank
        elif self.comm_type == CommunicationType.GRPC:
            return self.comm.rank
        else:
            raise NotImplementedError(
                "Rank not implemented for communication type", self.comm_type
            )

    def send(self, dest: str | int | List[str | int], data: Any, tag: int = 0):
        if isinstance(dest, list):
            for d in dest:
                self.comm.send(dest=int(d), data=data)
        else:
            print(f"Sending data to {dest}")
            self.comm.send(dest=int(dest), data=data)

    def receive(self, node_ids: str | int | List[str | int], tag: int = 0) -> Any:
        """
        Receive data from the specified node
        Returns a list if multiple node_ids are provided, else just returns the data
        """
        if isinstance(node_ids, list):
            items: List[Any] = []
            for id in node_ids:
                items.append(self.comm.receive(id))
            return items
        else:
            return self.comm.receive(node_ids)

    def broadcast(self, data: Any, tag: int = 0):
        self.comm.broadcast(data)

    def all_gather(self, tag: int = 0):
        return self.comm.all_gather()

    def finalize(self):
        self.comm.finalize()
