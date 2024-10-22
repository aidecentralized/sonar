from typing import Dict, Any, List
from mpi4py import MPI
from utils.communication.interface import CommunicationInterface
import threading
import time

class MPICommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]], data: Any):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Ensure that we are using thread safe threading level
        self.required_threading_level = MPI.THREAD_MULTIPLE
        self.threading_level = MPI.Query_thread()
        # Make sure to check for MPI_THREAD_MULTIPLE threading level to support 
        # thread safe calls to send and recv
        if self.required_threading_level > self.threading_level:
            raise RuntimeError(f"Insufficient thread support. Required: {self.required_threading_level}, Current: {self.threading_level}") 
        
        listener_thread = threading.Thread(target=self.listener, daemon=True)
        listener_thread.start()
        send_thread = threading.Thread(target=self.send, args=(data))
        send_thread.start()

        self.send_event = threading.Event()
        # Ensures that the listener thread and send thread are not using self.request_source at the same time
        self.source_node_lock = threading.Lock()
        self.request_source: int | None = None

    def initialize(self):
        pass

    def listener(self):
        """
        Runs on listener thread on each node to receive a send request
        Once send request is received, the listener thread informs the main
        thread to send the data to the requesting node.
        """
        while True:
            status = MPI.Status()
            # look for message with tag 1 (represents send request)
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=1, status=status):
                with self.source_node_lock:
                    self.request_source = status.Get_source()

                self.comm.irecv(source=self.request_source, tag=1)         
                self.send_event.set()
            time.sleep(1)  # Simulate waiting time 

    def send(self, data: Any):
        """
        Node will wait until request is received and then send
        data to requesting node.
        """
        while True:
            # Wait until the listener thread detects a request
            self.send_event.wait()
            with self.source_node_lock:
                dest = self.request_source

            if dest is not None:
                req = self.comm.isend(data, dest=int(dest))
                req.wait()
            
            with self.source_node_lock:
                self.request_source = None

            self.send_event.clear()

    def receive(self, node_ids: str | int) -> Any:
        """
        Node will send a request and wait to receive data.
        """
        node_ids = int(node_ids)
        send_req = self.comm.isend("", dest=node_ids, tag=1)
        send_req.wait()
        recv_req = self.comm.irecv(source=node_ids)
        return recv_req.wait()
    
    # depreciated broadcast function
    # def broadcast(self, data: Any):
    #     for i in range(1, self.size):
    #         if i != self.rank:
    #             self.send(i, data)

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

