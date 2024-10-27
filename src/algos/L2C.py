"""
This module defines the L2CClient and L2CServer classes, implementing a federated learning system with collaborative 
weight learning and local client pruning.
"""

from collections import defaultdict
from typing import Any

from utils.communication.comm_utils import CommunicationManager

import numpy as np

import torch
from torch import Tensor, optim
import torch.nn.functional as F

from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from algos.base_class import BaseFedAvgClient, BaseFedAvgServer


class L2CClient(BaseFedAvgClient):
    """
    Client class for L2C (Local Learning with Collaborative Weights).
    This class defines the client behavior in a federated learning system where clients share updates or weights,
    collaboratively learn weights, and prune neighbors based on collaborative weights.

    Attributes:
        config (dict): Configuration dictionary with settings.
        comm_utils (CommunicationManager): Utility for communication between clients and the server.
        alpha (Tensor): Tensor representing collaborative weights for each client.
        collab_weights (Tensor): Softmax-normalized collaborative weights.
        alpha_optim (optim.Optimizer): Optimizer for collaborative weights.
        sharing_mode (str): Sharing mode, either "updates" or "weights".
        neighbors_id_to_idx (dict): Mapping of neighbor IDs to indices.
    """

    def __init__(
        self, config: dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        """
        Initializes the L2CClient class.

        Args:
            config (dict): Configuration dictionary with settings.
            comm_utils (CommunicationManager): Communication manager for handling message passing.
        """
        super().__init__(config, comm_utils)
        self.init_collab_weights()
        self.sharing_mode: str = self.config["sharing"]
        self.neighbors_id_to_idx: dict[int, int] = {idx + 1: idx for idx in range(self.config["num_users"])}
        self.alpha: Tensor = torch.ones(self.config["num_users"], requires_grad=True)
        self.collab_weights: Tensor = F.softmax(self.alpha, dim=0)
        self.alpha_optim: optim.Optimizer = optim.Adam(
            [self.alpha],
            lr=self.config["alpha_lr"],
            weight_decay=self.config["alpha_weight_decay"],
        )

    def init_collab_weights(self) -> None:
        """
        Initialize collaborative weights for the current client. Collaborative weights represent the importance of 
        neighbor clients' updates in the federated learning process.
        """
        n: int = self.config["num_users"]
        # Neighbors id = [1, ..., num_users]
        # Neighbors idx = [0, ..., num_users - 1]
        self.neighbors_id_to_idx: dict[int, int] = {idx + 1: idx for idx in range(n)}

        # Initialize alpha and collaborative weights
        self.alpha: Tensor = torch.ones(n, requires_grad=True)
        self.collab_weights: Tensor = F.softmax(self.alpha, dim=0)
        self.alpha_optim: optim.Optimizer = optim.Adam(
            [self.alpha],
            lr=self.config["alpha_lr"],
            weight_decay=self.config["alpha_weight_decay"],
        )

    def filter_out_worse_neighbors(self, num_neighbors_to_keep: int) -> None:
        """
        Keep only the top k neighbors (+ itself) based on collaborative weights.

        Args:
            num_neighbors_to_keep (int): Number of top neighbors to retain.
        
        Raises:
            ValueError: If the number of neighbors to keep is greater than the total neighbors.
        """
        if num_neighbors_to_keep >= self.alpha.shape[0]:
            raise ValueError(
                "Number of neighbors to keep is greater than the number of neighbors"
            )

        if num_neighbors_to_keep <= 0:
            self.neighbors_id_to_idx = {self.node_id: 0}
            self.alpha = torch.ones(1, requires_grad=True)
        else:
            own_idx: int = self.neighbors_id_to_idx[self.node_id]
            own_mask: Tensor = torch.ones(self.alpha.shape, dtype=torch.bool)
            own_mask[own_idx] = 0
            remaining_neighbors_idx: list[int] = torch.topk(
                self.alpha[own_mask], k=num_neighbors_to_keep, largest=True
            )[1].tolist()

            remaining_neighbors_idx = [
                idx + 1 if idx >= own_idx else idx for idx in remaining_neighbors_idx
            ]
            remaining_neighbors_idx.append(own_idx)

            remaining_neighbors_id_idx: list[tuple[int, int]] = [
                (id, idx)
                for id, idx in self.neighbors_id_to_idx.items()
                if idx in remaining_neighbors_idx
            ]
            remaining_neighbors_id_idx = sorted(
                remaining_neighbors_id_idx, key=lambda x: x[1]
            )
            sorted_idx: list[int] = [idx for _, idx in remaining_neighbors_id_idx]
            self.alpha = self.alpha[sorted_idx].detach().requires_grad_(True)
            self.neighbors_id_to_idx = {
                id: new_idx
                for new_idx, (id, _) in enumerate(remaining_neighbors_id_idx)
            }

        self.collab_weights = F.softmax(self.alpha, dim=0)
        self.alpha_optim = optim.Adam(
            [self.alpha],
            lr=self.config["alpha_lr"],
            weight_decay=self.config["alpha_weight_decay"],
        )

    def learn_collab_weights(
        self, models_update_wts: dict[int, dict[str, Tensor]]
    ) -> tuple[float, float]:
        """
        Learn collaborative weights by backpropagating gradients from validation loss.

        Args:
            models_update_wts (dict): A dictionary containing models' updates/weights for neighbors.
        
        Returns:
            tuple: A tuple containing the alpha loss and accuracy.
        """
        self.model.eval()
        alpha_loss: float = 0
        correct: int = 0

        for data, target in self.val_dloader:
            data, target = data.to(self.device), target.to(self.device)
            self.alpha_optim.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()

            grad_dict: dict[str, Tensor] = {
                k: v.grad for k, v in self.model.named_parameters()
            }

            collab_weights_grads: list[Tensor] = []
            for i in self.neighbors_id_to_idx.keys():
                cw_grad: Tensor = torch.tensor(0.0)
                for key in grad_dict.keys():
                    if key not in self.model_keys_to_ignore:
                        if self.sharing_mode == "updates":
                            cw_grad -= (
                                models_update_wts[id][key] * grad_dict[key].cpu()
                            ).sum()
                        elif self.sharing_mode == "weights":
                            cw_grad += (
                                models_update_wts[id][key] * grad_dict[key].cpu()
                            ).sum()
                        else:
                            raise ValueError("Unknown sharing mode")
                collab_weights_grads.append(cw_grad)

            self.collab_weights.backward(torch.tensor(collab_weights_grads))
            self.alpha_optim.step()

            alpha_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.train()
        acc: float = correct / len(self.val_dloader.dataset)
        print(f"Node {self.node_id}'s updating alpha loss: {alpha_loss}, acc: {acc}")

        self.collab_weights = F.softmax(self.alpha, dim=0)
        return alpha_loss, acc

    def print_GPU_memory(self) -> None:
        """
        Print the GPU memory status for debugging purposes.
        """
        r: int = torch.cuda.memory_reserved(0)
        a: int = torch.cuda.memory_allocated(0)
        f: int = r - a  # free inside reserved
        print(
            f"Client {self.node_id} :GPU memory: reserved {r}, allocated {a}, free {f}"
        )

    def get_collaborator_weights(
        self, reprs_dict: dict[int, Tensor]
    ) -> dict[int, float]:
        """
        Get collaborative weights of the clients for the current round.

        Args:
            reprs_dict (dict): A dictionary containing representations of all clients.

        Returns:
            dict: A dictionary mapping client IDs to collaborative weights.
        """
        if self.node_id not in reprs_dict:
            raise ValueError("Own model not included in representations")

        collab_weights_dict: dict[int, float] = defaultdict(lambda: 0.0)
        for i, idx in self.neighbors_id_to_idx.items():
            collab_weights_dict[i] = self.collab_weights[idx]

        return collab_weights_dict

    def get_model_weights_without_keys_to_ignore(self) -> dict[str, Tensor]:
        """
        Get the current model's weights, excluding the weights that are marked to be ignored.

        Returns:
            dict: A dictionary of model weights with ignored keys removed.
        """
        return self.model_utils.filter_model_weights(
            self.get_model_weights(), self.model_keys_to_ignore
        )

    def get_representation(self) -> dict[str, Tensor]:
        """
        Get the current model's representation (either updates or weights).

        Returns:
            dict: The model's weights or updates, depending on the sharing mode.
        
        Raises:
            ValueError: If the sharing mode is unknown.
        """
        if self.sharing_mode == "updates":
            return self.model_utils.substract_model_weights(
                self.prev_model, self.get_model_weights_without_keys_to_ignore()
            )
        if self.sharing_mode == "weights":
            return self.get_model_weights()
        
        raise ValueError("Unknown sharing mode")

    def run_protocol(self) -> None:
        """
        Run the federated learning protocol for the client, including training, sharing representations, 
        and learning collaborative weights.
        """
        start_round: int = self.config.get("start_round", 0)
        total_rounds: int = self.config["rounds"]
        epochs_per_round: int = self.config["epochs_per_round"]

        for round_id in range(start_round, total_rounds):
            if self.sharing_mode == "updates":
                self.prev_model = self.get_model_weights_without_keys_to_ignore()

            cw: np.ndarray = np.zeros(self.config["num_users"])
            for i, idx in self.neighbors_id_to_idx.items():
                cw[i - 1] = self.collab_weights[idx]
            round_stats: dict[str, Any] = {"collab_weights": cw}

            self.comm_utils.receive(node_ids=self.server_node, tag=self.tag.ROUND_START)
            round_stats["train_loss"], round_stats["train_acc"] = self.local_train(
                epochs_per_round
            )
            repr: dict[str, Tensor] = self.get_representation()
            self.comm_utils.send(
                dest=self.server_node, data=repr, tag=self.tag.REPR_ADVERT
            )

            reprs: list[dict[str, Tensor]] = self.comm_utils.receive(
                node_ids=self.server_node, tag=self.tag.REPRS_SHARE
            )
            reprs_dict: dict[int, dict[str, Tensor]] = dict(enumerate(reprs, 1))

            collab_weights_dict: dict[int, float] = self.get_collaborator_weights(
                reprs_dict
            )
            models_update_wts: dict[int, dict[str, Tensor]] = reprs_dict

            new_wts: dict[str, Tensor] = self.aggregate(
                models_update_wts, collab_weights_dict, self.model_keys_to_ignore
            )

            if self.sharing_mode == "updates":
                new_wts = self.model_utils.substract_model_weights(
                    self.prev_model, new_wts
                )

            self.set_model_weights(new_wts, self.model_keys_to_ignore)
            round_stats["test_acc"] = self.local_test()
            round_stats["validation_loss"], round_stats["validation_acc"] = (
                self.learn_collab_weights(models_update_wts)
            )

            print(f"node {self.node_id} weight: {self.collab_weights}")
            if round_id == self.config["T_0"]:
                self.filter_out_worse_neighbors(self.config["target_users_after_T_0"])

            self.comm_utils.send(
                dest=self.server_node, data=round_stats, tag=self.tag.ROUND_STATS
            )


class L2CServer(BaseFedAvgServer):
    """
    Server class for L2C (Local Learning with Collaborative Weights).
    The server aggregates updates from clients, coordinates the federated learning process, and manages communication.

    Attributes:
        config (dict): Configuration dictionary with settings.
        model_save_path (str): Path to save the server's model.
    """

    def __init__(
        self, config: dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        """
        Initializes the L2CServer class.

        Args:
            config (dict): Configuration dictionary with settings.
            comm_utils (CommunicationManager): Communication manager for handling message passing.
        """
        super().__init__(config, comm_utils)
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path: str = (
            f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        )

    def test(self) -> float:
        """
        Test the model on the server.

        Returns:
            float: Accuracy of the model on the test set.
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        return acc

    def single_round(self) -> list[dict[str, Any]]:
        """
        Execute a single round of federated learning, including gathering updates from clients and sending new representations.

        Returns:
            list: A list of statistics from the clients for the current round.
        """
        for client_node in self.users:
            self.comm_utils.send(dest=client_node, data=None, tag=self.tag.ROUND_START)
        self.log_utils.log_console(
            "Server waiting for all clients to finish local training"
        )

        reprs: list[dict[str, Tensor]] = self.comm_utils.all_gather(
            self.tag.REPR_ADVERT
        )
        self.log_utils.log_console("Server received all clients models")
        self.send_representations(reprs)

        round_stats: list[dict[str, Any]] = self.comm_utils.all_gather(
            self.tag.ROUND_STATS
        )
        self.log_utils.log_console("Server received all clients stats")
        self.log_utils.log_tb_round_stats(round_stats, ["collab_weights"], self.round)
        self.log_utils.log_console(
            f"Round test acc {[stats['test_acc'] for stats in round_stats]}"
        )

        return round_stats

    def run_protocol(self) -> None:
        """
        Run the federated learning protocol on the server, coordinating multiple rounds of training and communication with clients.
        """
        self.log_utils.log_console("Starting L2C")
        start_round: int = self.config.get("start_round", 0)
        total_rounds: int = self.config["rounds"]

        stats: list[dict[str, Any]] = []
        for round_id in range(start_round, total_rounds):
            self.round = round_id
            self.log_utils.log_console(f"Starting round {round_id}")
            round_stats: list[dict[str, Any]] = self.single_round()
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)
