"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, OrderedDict
from utils.communication.comm_utils import CommunicationManager
import torch
import time

from algos.base_class import BaseFedAvgClient
from algos.topologies.collections import select_topology

from utils.gias import gia_main

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
        if "gia" in config:
            self.params_s = dict()
            self.params_t = dict()

            self.attack_start_round = 0
            self.attack_end_round = 1

            self.base_params = [key for key, _ in self.model.named_parameters()]

    def get_representation(self, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        """
        Returns the model weights as representation.
        """
        return self.get_model_weights()
    
    def receive_attack_and_aggregate(self, neighbors: List]) -> None:
        """
        Receives updates, launch GIA attack, and aggregate
        """
        for neighbor_info in neighbors:
            # Receive the attack from the neighbors
            neighbor_id = neighbor_info["sender"]
            neighbor_model = neighbor_info["model"]
            neighbor_images = neighbor_info["images"]
            neighbor_labels = neighbor_info["labels"]

            if round == self.attack_start_round:
                self.params_s[neighbor_id] = neighbor_model
            elif round == self.attack_end_round:
                self.params_t[neighbor_id] = neighbor_model

                # attack 
                p_s, p_t = self.params_s[neighbor_id], self.params_t[neighbor_id]
                gia_main(p_s, p_t, self.base_params, self.model, neighbor_labels, neighbor_images, self.node_id)


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
            self.local_round_done()

            # Collect the representations from all other nodes from the server
            neighbors = self.topology.sample_neighbours(self.num_collaborators)

            # TODO: Log the neighbors
            stats["neighbors"] = neighbors

            if "gia" in self.config:
                if it < self.attack_start_round or it > self.attack_end_round:
                    self.receive_and_aggregate(neighbors)

                else:
                    self.receive_attack_and_aggregate(neighbors)
            else:
                self.receive_and_aggregate(neighbors)

            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()

            # evaluate the model on the test data
            # Inside FedStaticNode.run_protocol()
            stats["test_loss"], stats["test_acc"] = self.local_test()
            self.log_metrics(stats=stats, iteration=it)

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
