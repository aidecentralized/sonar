"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, OrderedDict, List
from collections import OrderedDict, defaultdict

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
            if int(self.node_id) in self.config["gia_attackers"]:
                self.gia_attacker = True
            self.params_s = dict()
            self.params_t = dict()
            # Track neighbor updates with a dictionary mapping neighbor_id to their updates
            self.neighbor_updates = defaultdict(list)
            # Track which neighbors we've already attacked
            self.attacked_neighbors = set()

            self.base_params = [key for key, _ in self.model.named_parameters()]

    def get_representation(self, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        """
        Returns the model weights as representation.
        """
        return self.get_model_weights()
    
    def receive_attack_and_aggregate(self, neighbors: List[int], round: int, num_neighbors: int) -> None:
        """
        Receives updates, launches GIA attack when second update is seen from a neighbor
        """
        print("CLIENT RECEIVING ATTACK AND AGGREGATING")
        if self.is_working:
            # Receive the model updates from the neighbors
            model_updates = self.comm_utils.receive(node_ids=neighbors)
            assert len(model_updates) == num_neighbors

            for neighbor_info in model_updates:
                neighbor_id = neighbor_info["sender"]
                neighbor_model = neighbor_info["model"]
                neighbor_model = OrderedDict(
                    (key, value) for key, value in neighbor_model.items()
                    if key in self.base_params
                )

                neighbor_images = neighbor_info["images"]
                neighbor_labels = neighbor_info["labels"]

                # Store this update
                self.neighbor_updates[neighbor_id].append({
                    "model": neighbor_model,
                    "images": neighbor_images,
                    "labels": neighbor_labels
                })

                # Check if we have 2 updates from this neighbor and haven't attacked them yet
                if len(self.neighbor_updates[neighbor_id]) == 2 and neighbor_id not in self.attacked_neighbors:
                    print(f"Client {self.node_id} attacking {neighbor_id}!")
                    
                    # Get the two parameter sets for the attack
                    p_s = self.neighbor_updates[neighbor_id][0]["model"]
                    p_t = self.neighbor_updates[neighbor_id][1]["model"]
                    
                    # Launch the attack
                    gia_main(
                        p_s, 
                        p_t, 
                        self.base_params, 
                        self.model, 
                        neighbor_labels, 
                        neighbor_images, 
                        self.node_id
                    )
                    
                    # Mark this neighbor as attacked
                    self.attacked_neighbors.add(neighbor_id)
                    
                    # Optionally, clear the stored updates to save memory
                    del self.neighbor_updates[neighbor_id]

            self.aggregate(model_updates, keys_to_ignore=self.model_keys_to_ignore)

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
            stats["neighbors"] = neighbors

            if hasattr(self, "gia_attacker"):
                print(f"Client {self.node_id} is a GIA attacker!")
                self.receive_attack_and_aggregate(neighbors, it, len(neighbors))
            else:
                self.receive_and_aggregate(neighbors)

            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()
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
