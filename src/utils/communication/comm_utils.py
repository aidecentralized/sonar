from abc import ABC, abstractmethod

class CommunicationInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def send(self, dest, data):
        pass

    @abstractmethod
    def receive(self, node_ids, data):
        pass

    @abstractmethod
    def broadcast(self, data):
        pass

    @abstractmethod
    def finalize(self):
        pass
