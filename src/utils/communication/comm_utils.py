from enum import Enum
from utils.communication.grpc.main import GRPCCommunication
from typing import Any, Dict, List, TYPE_CHECKING
from utils.communication.rtc4 import RTCCommUtils
import asyncio

if TYPE_CHECKING:
    from algos.base_class import BaseNode

class CommunicationType(Enum):
    MPI = 1
    GRPC = 2
    HTTP = 3
    RTC = 4

class CommunicationFactory:
    @staticmethod
    def create_communication(config: Dict[str, Any], comm_type: CommunicationType):
        if comm_type == CommunicationType.MPI:
            raise NotImplementedError("MPI's new version not yet implemented. Please use GRPC.")
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
        if "comm" not in config or "type" not in config["comm"]:
            raise KeyError("Missing 'comm' or 'type' in config")
        
        try:
            self.comm_type = CommunicationType[config["comm"]["type"].upper()]
        except KeyError:
            raise ValueError(f"Invalid communication type: {config['comm']['type']}. Must be one of {list(CommunicationType.__members__.keys())}")
        self.comm = CommunicationFactory.create_communication(config, self.comm_type)
        self.comm.initialize()
        self._ready = asyncio.Event()
    
    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()
        
    async def setup(self) -> None:
        try:
            success = await self.comm.initialize()
            if success:
                self._ready.set()
            else:
                raise RuntimeError("Communication initialization failed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize communication: {e}")
            
    async def ensure_ready(self) -> None:
        await self._ready.wait()

    def register_node(self, obj: "BaseNode"):
        self.comm.register_self(obj)

    def get_rank(self) -> int:
        print(f"[DEBUG] Checking rank assignment, current type: {self.comm_type}")
        
        if self.comm_type == CommunicationType.RTC:
            if hasattr(self.comm, "rank"):
                print(f"[DEBUG] RTC rank assigned: {self.comm.rank}")
                if self.comm.rank is None:
                    raise ValueError("[ERROR] RTC rank assignment failed.")
                return self.comm.rank
            else:
                raise ValueError("[ERROR] RTC Communication Manager has no rank attribute.")
        
        raise NotImplementedError("[ERROR] Rank retrieval not implemented for this communication type")



    def send(self, dest: str | int | List[str | int], data: Any, tag: int = 0):
        if isinstance(dest, list):
            for d in dest:
                self.comm.send(dest=int(d), data=data)
        else:
            self.comm.send(dest=int(dest), data=data)

    def receive(self, node_ids: List[int]) -> Any:
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
