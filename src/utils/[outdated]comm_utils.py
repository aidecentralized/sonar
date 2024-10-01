from mpi4py import MPI
from typing import Any, Optional, List


class CommUtils:
    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def send_signal(self, dest: int, data: Any, tag: Optional[int] = None) -> None:
        if tag is not None:
            self.comm.send(data, dest=dest, tag=tag)
        else:
            self.comm.send(data, dest=dest)

    def send_signal_to_all_clients(self, client_ids: List[int], data: Any, tag: Optional[int] = None) -> None:
        for client_id in client_ids:
            self.send_signal(client_id, data, tag=tag)

    def wait_for_signal(self, src: int, tag: Optional[int] = None) -> Any:
        """
        Wait for a signal from a source node
        """
        if tag is not None:
            recv_data = self.comm.recv(source=src, tag=tag)
        else:
            recv_data = self.comm.recv(source=src)
        return recv_data

    def wait_for_all_clients(self, client_ids: List[int], tag: Optional[int] = None) -> List[Any]:
        """
        Wait for signals from all client nodes
        """
        data_list = []
        for client_id in client_ids:
            data = self.wait_for_signal(client_id, tag=tag)
            data_list.append(data)
        return data_list