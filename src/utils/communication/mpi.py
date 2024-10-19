from typing import Dict, Any, List
from mpi4py import MPI
from utils.communication.interface import CommunicationInterface
import threading
import time
from enum import Enum


class MPICommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def initialize(self):
        pass

    # def send(self, dest: str | int, data: Any):
    #     self.comm.send(data, dest=int(dest))

    # def receive(self, node_ids: str | int) -> Any:
    #     return self.comm.recv(source=int(node_ids))

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


class MPICommunication(MPICommUtils):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        super().__init__(config)
        listener_thread = threading.Thread(target=self.listener, daemon=True)
        listener_thread.start()
        self.send_event = threading.Event()
        self.request_source: int | None = None

    def listener(self):
        while True:
            status = MPI.Status()
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
                source = status.Get_source()
                tag = status.Get_tag()
                count = status.Get_count(MPI.BYTE)  # Get the number of bytes in the message
                # If a message is available, receive it
                data_to_recv = bytearray(count)
                req = self.comm.irecv([data_to_recv, MPI.BYTE], source=source, tag=tag)         
                req.wait()   
                # Convert the byte array back to a string
                received_message = data_to_recv.decode('utf-8')
                
                if received_message == "Requesting Information":
                    self.send_event.set()

                self.send_event.clear()
                break
            time.sleep(1)  # Simulate waiting time 

    def send(self, dest: str | int, data: Any, tag: int):
        while True:
            # Wait until the listener thread detects a request
            self.send_event.wait()
            req = self.comm.isend(data, dest=int(dest), tag=tag)
            req.wait()

    def receive(self, node_ids: str | int, tag: int) -> Any:
        node_ids = int(node_ids)
        message = "Requesting Information"
        message_bytes = bytearray(message, 'utf-8')
        send_req = self.comm.isend([message_bytes, MPI.BYTE], dest=node_ids, tag=tag)
        send_req.wait()
        recv_req = self.comm.irecv(source=node_ids, tag=tag)
        return recv_req.wait()

# MPI Server
"""
initialization():
    node spins up listener thread, threading (an extra thread might not be needed since iprobe exists).
    call listen?

listen():
    listener thread starts listening for send requests (use iprobe and irecv for message)
    when send request is received, call the send() function

send():
    gather and send info to requesting node using comm.isend
    comm.wait
    
"""

# MPI Client
"""
initialization():
    node is initialized

receive():
    node sends request to sending node using isend()
    node calls irecv and waits for response
"""

