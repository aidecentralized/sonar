from enum import Enum
from typing import Any, Dict

from utils.communication.grpc.main import GRPCCommunication
from utils.communication.mpi import MPICommUtils


class CommunicationType(Enum):
    MPI = 1
    GRPC = 2
    HTTP = 3


class CommunicationFactory:
    @staticmethod
    def create_communication(config: Dict[str, Any], comm_type: CommunicationType) -> Any:
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
            raise NotImplementedError("Rank not implemented for communication type", self.comm_type)

    def send(self, dest:str|int, data:Any):
        self.comm.send(dest, data)

    def receive(self, node_ids: str|int) -> Any:
        return self.comm.receive(node_ids)

    def broadcast(self, data: Any):
        self.comm.broadcast(data)

    def all_gather(self):
        return self.comm.all_gather()

    def finalize(self):
        self.comm.finalize()
