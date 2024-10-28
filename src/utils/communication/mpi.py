from typing import Dict, Any, List, TYPE_CHECKING
from mpi4py import MPI
from torch import Tensor
from utils.communication.interface import CommunicationInterface
import threading
import time
from utils.communication.grpc.grpc_utils import deserialize_model, serialize_model
import random

if TYPE_CHECKING:
    from algos.base_class import BaseNode

class MPICommUtils(CommunicationInterface):
    def __init__(self, config: Dict[str, Dict[str, Any]]):
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
        
        self.send_event = threading.Event()
        # Ensures that the listener thread and send thread are not using self.request_source at the same time
        self.lock = threading.Lock()
        self.request_source: int | None = None

        self.is_working = True
        self.communication_cost_received: int = 0
        self.communication_cost_sent: int = 0

        self.base_node: BaseNode | None = None
        
        listener_thread = threading.Thread(target=self.listener, daemon=True)
        listener_thread.start()

    def initialize(self):
        pass

    def register_self(self, obj: "BaseNode"):
        self.base_node = obj
        send_thread = threading.Thread(target=self.send)
        send_thread.start()
    
    def get_comm_cost(self):
        with self.lock:
            return self.communication_cost_received, self.communication_cost_sent

    def listener(self):
        """
        Runs on listener thread on each node to receive a send request
        Once send request is received, the listener thread informs the send
        thread to send the data to the requesting node.
        """
        while True:
            status = MPI.Status()
            # look for message with tag 1 (represents send request)
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=1, status=status):
                with self.lock:
                    self.request_source = status.Get_source()

                self.comm.irecv(source=self.request_source, tag=1)         
                self.send_event.set()
            time.sleep(1)  # Simulate waiting time 

    def get_model(self) -> bytes | None:
        print(f"getting model from {self.rank}, {self.base_node}")
        if not self.base_node:
            raise Exception("Base node not registered")
        with self.lock:
            if self.is_working:
                print("model is working")
                model = serialize_model(self.base_node.get_model_weights())
                print(f"model data to be sent: {model}")
            else:
                assert self.base_node.dropout.dropout_enabled, "Empty models are only supported when Dropout is enabled."
                model = None
            return model
    
    def send(self):
        """
        Node will wait for a request to send data and then send the
        data to requesting node.
        """
        while True:
            # Wait until the listener thread detects a request
            self.send_event.wait()
            with self.lock:
                dest = self.request_source

            if dest is not None:
                data = self.get_model()
                req = self.comm.isend(data, dest=int(dest))
                req.wait()
            
            with self.lock:
                self.request_source = None

            self.send_event.clear()

    def receive(self, node_ids: List[int]) -> Any:
        """
        Node will send a request for data and wait to receive data.
        """
        max_tries = 10
        for node in node_ids:
            while max_tries > 0:
                try:
                    self.comm.send("", dest=node, tag=1)
                    recv_req = self.comm.irecv(source=node)
                    received_data = recv_req.wait()
                    print(f"received data: {received_data}")
                    return deserialize_model(received_data)
                except Exception as e:
                    print(f"MPI failed {10 - max_tries} times: {e}", "Retrying...")
                    import traceback
                    print(traceback.print_exc())
                    # sleep for a random time between 1 and 10 seconds
                    random_time = random.randint(1, 10)
                    time.sleep(random_time)
                    max_tries -= 1
    
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
        # 1. All nodes send finished to the super node
        # 2. super node will wait for all nodes to send finished
        # 3. super node will then send bye to all nodes
        # 4. all nodes will wait for the bye and then exit
        # this is to ensure that all nodes have finished
        # and no one leaves early
        if self.rank == 0:
            quorum_threshold = self.num_users - 1 # No +1 for the super node because it doesn't send finished
            num_finished: set[int] = set() 
            status = MPI.Status()
            while len(num_finished) < quorum_threshold:
                print(
                    f"Waiting for {quorum_threshold} users to finish, {num_finished} have finished so far"
                )
                # get finished nodes
                self.comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
                print(f"received finish message from {status.Get_source()}")
                num_finished.add(status.Get_source())
                
        else:
            # send finished to the super node
            print(f"Node {self.rank} sent finish message")
            self.send_finished()
        
        message = self.comm.bcast("Done", root=0)
        self.finished = True
        self.send_event.set()
        print(f"Node {self.rank} received {message}, finished")
        self.comm.Barrier()
        self.listener_thread.join()
        print(f"Node {self.rank} listener thread done")
        print(f"Node {self.rank} listener thread is {self.listener_thread.is_alive()}")
        print(f"Node {self.rank} {threading.enumerate()}")
        self.comm.Barrier()
        print(f"Node {self.rank}: all nodes synchronized")
        MPI.Finalize()
    
    def set_is_working(self, is_working: bool):
        with self.lock:
            self.is_working = is_working

    def set_is_working(self, is_working: bool):
        with self.lock:
            self.is_working = is_working

