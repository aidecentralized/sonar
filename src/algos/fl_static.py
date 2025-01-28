"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""

from typing import Any, Dict, OrderedDict, List
from utils.communication.comm_utils import CommunicationManager
import torch

from algos.base_class import BaseFedAvgClient
from algos.topologies.collections import select_topology


class FedStaticNode(BaseFedAvgClient):
    """
    Federated Static Client Class.
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)

        # Keep the original topology setup:
        self.topology = select_topology(config, self.node_id)
        self.topology.initialize()

        # Server-based neighbors (if set later via store_server_neighbors)
        self.server_neighbors: List[int] = []

    def store_server_neighbors(self, neighbor_dict: Dict[str, int]) -> None:
        """
        Call this method after receiving a 'topology' message from the signaling server.
        For instance, if the server sends {"prev": 0, "next": 2},
        we convert it to a list [0, 2].
        """
        self.server_neighbors = list(neighbor_dict.values())

    def get_neighbors(self) -> List[int]:
        """
        Returns a list of neighbours for the client.
        
        1) If server-based neighbors are available, use them.
        2) Otherwise, fallback to the local topology from select_topology.
        """
        # If the signaling server hasn't provided ring neighbors yet, 
        # fall back to local sampling
        if not self.server_neighbors:
            neighbors = self.topology.sample_neighbours(self.num_collaborators)
            self.stats["neighbors"] = neighbors
            return neighbors

        # Otherwise, use server-based neighbors
        self.stats["neighbors"] = self.server_neighbors
        return self.server_neighbors

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError(
                "Start round different from 0 not implemented yet"
            )
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)
        for it in range(start_round, total_rounds):
            self.round_init()

            # Train locally and send the representation to the server
            self.local_train(it, epochs_per_round)
            self.local_round_done()

            # Get neighbors (could be server-based ring or fallback local)
            neighbors = self.get_neighbors()
            self.receive_and_aggregate(neighbors)

            # Evaluate the model on test data
            self.local_test()
            self.round_finalize()

    async def run_async_protocol(self) -> None:
        """
        Asynchronous version of run_protocol
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError(
                "Start round different from 0 not implemented yet"
            )
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)
        for it in range(start_round, total_rounds):
            self.round_init()

            # Train locally and send the representation to the server
            self.local_train(it, epochs_per_round)
            self.local_round_done()

            # Get neighbors (could be server-based ring or fallback local)
            neighbors = self.get_neighbors()
            await self.receive_and_aggregate_async(neighbors)

            # Evaluate the model
            self.local_test()
            self.round_finalize()


class FedStaticServer(BaseFedAvgClient):
    """
    Federated Static Server Class. It does not do anything.
    It just exists to keep the code compatible across different algorithms.
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        pass

    def run_protocol(self) -> None:
        pass