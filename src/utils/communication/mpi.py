from collections import OrderedDict
from typing import Dict, Any, List, TYPE_CHECKING
from mpi4py import MPI
from torch import Tensor
from utils.communication.interface import CommunicationInterface
import threading
import time
import random
import numpy as np

if TYPE_CHECKING:
    from algos.base_class import BaseNode

class MPICommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.num_users: int = int(config["num_users"])  # type: ignore
        self.finished = False

        # Ensure that we are using thread safe threading level
        self.required_threading_level = MPI.THREAD_MULTIPLE
        self.threading_level = MPI.Query_thread()
        # Make sure to check for MPI_THREAD_MULTIPLE threading level to support 
        # thread safe calls to send and recv
        if self.required_threading_level > self.threading_level:
            raise RuntimeError(f"Insufficient thread support. Required: {self.required_threading_level}, Current: {self.threading_level}") 
        
        self.send_event = threading.Event()
        # Ensures that the listener thread and send thread are not using self.request_source at the same time
        self.lock = threading.Lock()
        self.request_source: int | None = None

        self.is_working = True
        self.communication_cost_received: int = 0
        self.communication_cost_sent: int = 0

        self.base_node: BaseNode | None = None
        
        self.listener_thread = threading.Thread(target=self.listener)
        self.listener_thread.start()

        self.send_thread = threading.Thread(target=self.send)

    def initialize(self):
        pass

    def register_self(self, obj: "BaseNode"):
        self.base_node = obj
        self.send_thread.start()
    
    def get_comm_cost(self):
        with self.lock:
            return self.communication_cost_received, self.communication_cost_sent

    def listener(self):
        """
        Runs on listener thread on each node to receive a send request
        Once send request is received, the listener thread informs the send
        thread to send the data to the requesting node.
        """
        while not self.finished:
            status = MPI.Status()
            # look for message with tag 1 (represents send request)
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=1, status=status):
                with self.lock:
                    self.request_source = status.Get_source()

                print(f"Node {self.rank} received request from {self.request_source}")
                # receive_request = self.comm.irecv(source=self.request_source, tag=1)  
                # receive_request.wait()
                self.comm.recv(source=self.request_source, tag=1)
                self.send_event.set()
            # time.sleep(1) 
        print(f"Node {self.rank} listener thread ended")

    def get_model(self) -> List[OrderedDict[str, Tensor]] | None:
        print(f"getting model from {self.rank}, {self.base_node}")
        if not self.base_node:
            raise Exception("Base node not registered")
        with self.lock:
            if self.is_working:
                model = self.base_node.get_model_weights()
                model = [model]
                print(f"Model from {self.rank} acquired")
            else:
                assert self.base_node.dropout.dropout_enabled, "Empty models are only supported when Dropout is enabled."
                model = None
            return model
    
    def send(self):
        """
        Node will wait for a request to send data and then send the
        data to requesting node.
        """
        while not self.finished:
            # Wait until the listener thread detects a request
            self.send_event.wait()
            if self.finished:
                break
            with self.lock:
                dest = self.request_source

            if dest is not None:
                data = self.get_model()
                print(f"Node {self.rank} is sending data to {dest}")
                # req = self.comm.Isend(data, dest=int(dest))
                # req.wait()
                self.comm.send(data, dest=int(dest))
            
            with self.lock:
                self.request_source = None

            self.send_event.clear()
        print(f"Node {self.rank} send thread ended")

    def receive(self, node_ids: List[int]) -> Any:
        """
        Node will send a request for data and wait to receive data.
        """
        max_tries = 10
        assert len(node_ids) == 1, "Too many node_ids to unpack"
        node = node_ids[0]
        while max_tries > 0:
            try:
                print(f"Node {self.rank} receiving from {node}")
                self.comm.send("", dest=node, tag=1)
                # recv_req = self.comm.Irecv([], source=node)
                # received_data = recv_req.wait()
                received_data = self.comm.recv(source=node)
                print(f"Node {self.rank} received data from {node}: {bool(received_data)}")
                if not received_data:
                    raise Exception("Received empty data")
                return received_data
            except MPI.Exception as e:
                print(f"MPI failed {10 - max_tries} times: MPI ERROR: {e}", "Retrying...")
                import traceback
                print(f"Traceback: {traceback.print_exc()}")
                # sleep for a random time between 1 and 10 seconds
                random_time = random.randint(1, 10)
                time.sleep(random_time)
                max_tries -= 1
            except Exception as e:
                print(f"MPI failed {10 - max_tries} times: {e}", "Retrying...")
                import traceback
                print(f"Traceback: {traceback.print_exc()}")
                # sleep for a random time between 1 and 10 seconds
                random_time = random.randint(1, 10)
                time.sleep(random_time)
                max_tries -= 1
        print(f"Node {self.rank} received")
    
    # deprecated broadcast function
    def broadcast(self, data: Any):
        for i in range(1, self.size):
            if i != self.rank:
                self.comm.send(data, dest=i)

    def all_gather(self):
        """
        This function is used to gather data from all the nodes.
        """
        items: List[Any] = []
        for i in range(1, self.size):
            print(f"receiving this data: {self.receive(i)}")
            print(f"receiving this data: {self.receive(i)}")
            items.append(self.receive(i))
        return items
    
    def send_finished(self):
        self.comm.send("Finished", dest=0, tag=2)

    def finalize(self):
        pass

    def set_is_working(self, is_working: bool):
        with self.lock:
            self.is_working = is_working

