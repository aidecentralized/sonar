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
        self.topology.initialize()

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
        epochs_per_round = self.config.get("epochs_per_round", 1)
        for it in range(start_round, total_rounds):
            # Train locally and send the representation to the server
            stats["train_loss"], stats["train_acc"] = self.local_train(
                epochs_per_round
            )
            self.local_round_done()

            # Collect the representations from all other nodes from the server
            neighbors = self.topology.sample_neighbours(self.num_collaborators)
            # TODO: Log the neighbors

            # Pull the model updates from the neighbors
            model_updates = self.comm_utils.receive(node_ids=neighbors)

            # Aggregate the representations
            self.aggregate(model_updates)

            # evaluate the model on the test data
            stats["test_loss"], stats["test_acc"] = self.local_test()
            self.log_utils.log_console("Round {} done for Node {}, stats {}".format(it, self.node_id, stats))
            self.log_utils.log_tb(key="train/loss", value=stats["train_loss"], iteration=it)
            self.log_utils.log_tb(key="train/accuracy", value=stats["train_acc"], iteration=it)
            self.log_utils.log_tb(key="test/loss", value=stats["test_loss"], iteration=it)
            self.log_utils.log_tb(key="test/accuracy", value=stats["test_acc"], iteration=it)


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