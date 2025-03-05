"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, OrderedDict, List
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
        assert self.streaming_aggregation == False, "Streaming aggregation not supported for push-based algorithms for now."

    def get_neighbors(self) -> List[int]:
        """
        Returns a list of neighbours for the client.
        """
        neighbors = self.topology.sample_neighbours(self.num_collaborators, mode="push")
        self.stats["neighbors"] = neighbors

        return neighbors

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
            self.local_train(
                    it, epochs_per_round
                )            

            # Collect the representations from all other nodes from the server
            neighbors = self.get_neighbors()
            self.push(neighbors)
            self.receive_pushed_and_aggregate()
            # evaluate the model on the test data
            # Inside FedStaticNode.run_protocol()
            self.local_test()
            self.local_round_done()

            self.round_finalize()



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
