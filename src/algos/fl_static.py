"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, OrderedDict
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
        self.topology = select_topology(config, self.node_id)

    def get_representation(self, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        """
        Returns the model weights as representation.
        """
        return self.get_model_weights()

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        stats: Dict[str, float] = {}
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError(
                "Start round different from 0 not implemented yet"
            )
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        for _ in range(start_round, total_rounds):
            # Train locally and send the representation to the server
            stats["train_loss"], stats["train_acc"] = self.local_train(
                epochs_per_round
            )

            # Collect the representations from all other nodes from the server
            neighbors = self.topology.sample_neighbours(self.num_collaborators)
            # TODO: Log the neighbors

            # Pull the model updates from the neighbors
            model_updates = self.comm_utils.receive(node_ids=neighbors)

            # Aggregate the representations
            self.aggregate(model_updates)

            # evaluate the model on the test data
            stats["test_loss"], stats["test_acc"] = self.local_test()
