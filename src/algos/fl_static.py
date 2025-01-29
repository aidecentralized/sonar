"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""

from typing import Any, Dict, List
from utils.communication.comm_utils import CommunicationManager
import torch

from algos.base_class import BaseFedAvgClient
from algos.topologies.collections import select_topology


class FedStaticNode(BaseFedAvgClient):
    def __init__(self, config, comm_utils):
        super().__init__(config, comm_utils)

        if self.node_id is None:
            print("RTC rank assignment failed, setting a default rank 0")
            self.node_id = 0

    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        self.topology = select_topology(config, self.node_id)  # Get topology from config
        self.topology.generate_graph()

    def get_neighbors(self) -> List[int]:
        """
        Returns a list of neighbors for the client based on topology.
        """
        if not hasattr(self.topology, "graph") or self.topology.graph is None:
            return []
        
        self.server_neighbors = list(self.topology.graph.neighbors(self.node_id))  # Retrieve neighbors from graph
        self.stats["neighbors"] = self.server_neighbors
        return self.server_neighbors

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError("Start round different from 0 not implemented yet")

        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)

        for it in range(start_round, total_rounds):
            self.round_init()
            self.local_train(it, epochs_per_round)
            self.local_round_done()

            neighbors = self.get_neighbors()
            self.receive_and_aggregate(neighbors)

            self.local_test()
            self.round_finalize()

    async def run_async_protocol(self) -> None:
        """
        Asynchronous version of run_protocol
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError("Start round different from 0 not implemented yet")

        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)

        for it in range(start_round, total_rounds):
            self.round_init()
            self.local_train(it, epochs_per_round)
            self.local_round_done()

            neighbors = self.get_neighbors()
            await self.receive_and_aggregate_async(neighbors)

            self.local_test()
            self.round_finalize()


class FedStaticServer(BaseFedAvgClient):
    """
    Federated Static Server Class. It does not do anything.
    """

    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        pass

    def run_protocol(self) -> None:
        pass
