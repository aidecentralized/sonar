from abc import ABC, abstractmethod
from typing import Any

class CommunicationInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def send(self, dest: str|int, data: Any):
        pass

    @abstractmethod
    def receive(self, node_ids: str|int) -> Any:
        pass

    @abstractmethod
    def broadcast(self, data: Any):
        pass

    @abstractmethod
    def all_gather(self) -> Any:
        pass

    @abstractmethod
    def finalize(self):
        pass
