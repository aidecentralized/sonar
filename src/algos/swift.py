"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, OrderedDict
from utils.communication.comm_utils import CommunicationManager
import torch
import time

from algos.fl_static import FedStaticNode, FedStaticServer
from algos.topologies.collections import select_topology


class SwiftNode(FedStaticNode):
    """
    Federated Static Client Class.
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        stats: Dict[str, Any] = {}
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError(
                "Start round different from 0 not implemented yet"
            )
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)
        for it in range(start_round, total_rounds):
            # Train locally and send the representation to the server
            stats["train_loss"], stats["train_acc"], stats["train_time"] = self.local_train(
                    it, epochs_per_round
                )            

            # Collect the representations from all other nodes from the server
            neighbors = self.topology.sample_neighbours(self.num_collaborators)
            # TODO: Log the neighbors
            stats["neighbors"] = neighbors

            self.push(neighbors)

            self.receive_pushed_and_aggregate()

            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()

            # evaluate the model on the test data
            # Inside FedStaticNode.run_protocol()
            stats["test_loss"], stats["test_acc"] = self.local_test()

            stats.update(self.get_memory_metrics())

            self.log_metrics(stats=stats, iteration=it)
            self.local_round_done()



class SwiftServer(FedStaticServer):
    """
    Swift Server Class. It does not do anything.
    It just exists to keep the code compatible across different algorithms.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        pass

    def run_protocol(self) -> None:
        pass
