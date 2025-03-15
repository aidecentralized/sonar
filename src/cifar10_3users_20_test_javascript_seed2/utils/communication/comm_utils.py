from enum import Enum
from utils.communication.grpc.main import GRPCCommunication
from typing import Any, Dict, List, TYPE_CHECKING
from utils.communication.rtc_async_ver import RTCCommUtils
# from utils.communication.mpi import MPICommUtils
import asyncio
import threading

if TYPE_CHECKING:
    from algos.base_class import BaseNode


class CommunicationType(Enum):
    MPI = 1
    GRPC = 2
    HTTP = 3
    RTC = 4


class CommunicationFactory:
    @staticmethod
    def create_communication(
        config: Dict[str, Any], comm_type: CommunicationType
    ):
        comm_type = comm_type
        if comm_type == CommunicationType.MPI:
            raise NotImplementedError("MPI's new version not yet implemented. Please use GRPC. See https://github.com/aidecentralized/sonar/issues/96 for more details.")
        elif comm_type == CommunicationType.GRPC:
            return GRPCCommunication(config)
        elif comm_type == CommunicationType.HTTP:
            raise NotImplementedError("HTTP communication not yet implemented")
        elif comm_type == CommunicationType.RTC:
            return RTCCommUtils(config)
        else:
            raise ValueError("Invalid communication type", comm_type)


class CommunicationManager:
    def __init__(self, config: Dict[str, Any]):
        self.comm_type = CommunicationType[config["comm"]["type"]]
        self.comm = CommunicationFactory.create_communication(config, self.comm_type)
        # webrtc_thread = threading.Thread(target=self.comm.initialize)
        # webrtc_thread.start()
        self.comm.initialize()
        # self._ready = asyncio.Event()
    
    #################################################
    @property
    def is_ready(self) -> bool:
        """Check if communication manager is ready."""
        return self._ready.is_set()
        
    async def setup(self) -> None:
        """
        Async setup method to initialize communication.
        This must be called after creating the CommunicationManager instance.
        """
        try:
            success = await self.comm.initialize()
            if success:
                self._ready.set()
            else:
                raise RuntimeError("Communication initialization failed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize communication: {e}")
            
    async def ensure_ready(self) -> None:
        """Wait for communication manager to be ready."""
        await self._ready.wait()
    #################################################

    def register_node(self, obj: "BaseNode"):
        self.comm.register_self(obj)

    def get_rank(self) -> int:
        if self.comm_type == CommunicationType.MPI:
            if self.comm.rank is None:
                raise ValueError("Rank not set for MPI")
            return self.comm.rank
        elif self.comm_type == CommunicationType.GRPC:
            if self.comm.rank is None:
                raise ValueError("Rank not set for gRPC")
            return self.comm.rank
        elif self.comm_type == CommunicationType.RTC:
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

    def receive(self, node_ids: List[int]) -> Any:
        """
        Receive data from the specified node
        Returns a list if multiple node_ids are provided, else just returns the data
        """
        return self.comm.receive(node_ids)

    def broadcast(self, data: Any, tag: int = 0):
        self.comm.broadcast(data)

    def all_gather(self, tag: int = 0):
        return self.comm.all_gather()

    def finalize(self):
        self.comm.finalize()

    def set_is_working(self, is_working: bool):
        self.comm.set_is_working(is_working)

    def get_comm_cost(self):
        return self.comm.get_comm_cost()

    def receive_pushed(self):
        return self.comm.receive_pushed()

    def all_gather_pushed(self):
        return self.comm.all_gather_pushed()