"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""

from collections import defaultdict
from typing import Any, Dict, List
from utils.communication.comm_utils import CommunicationManager
import numpy as np
import torch
import torch.nn as nn

from algos.base_class import BaseFedAvgClient, BaseFedAvgServer
from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from algos.fl_ring import RingTopology
from algos.fl_grid import GridTopology
from algos.fl_torus import TorusTopology
from algos.fl_random import RandomTopology


class FedStaticClient(BaseFedAvgClient):
    """
    Federated Static Client Class.
    """
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)

    def get_collaborator_weights(self, reprs_dict: Dict[int, Any], rnd: int) -> Dict[int, float]:
        """
        Returns the weights of the collaborators for the current round.
        """
        total_rounds = self.config["rounds"]
        within_community_sampling = self.config.get("within_community_sampling", 1)
        p_within_decay = self.config.get("p_within_decay", None)

        if p_within_decay is not None:
            within_community_sampling = self._decay_within_sampling(
                p_within_decay, within_community_sampling, rnd, total_rounds
            )

        algo = self.config["topology"]
        selected_ids = self._select_ids_based_on_algo(algo)

        collab_weights = defaultdict(lambda: 0.0)
        for idx in selected_ids:
            own_aggr_weight = self.config.get("own_aggr_weight", 1 / len(selected_ids))
            own_aggr_weight = self._apply_aggr_weight_strategy(
                own_aggr_weight, rnd, total_rounds
            )

            collab_weights[idx] = self._calculate_collab_weight(idx, own_aggr_weight, selected_ids)

        return collab_weights

    def _decay_within_sampling(self, strategy: str, p: float, rnd: int, total_rounds: int) -> float:
        """
        Applies the within-community sampling decay strategy.
        """
        if strategy == "linear_inc":
            p *= (rnd / total_rounds)
        elif strategy == "linear_dec":
            p *= (1 - rnd / total_rounds)
        elif strategy == "exp_inc":
            alpha = np.log((1 - p) / p)
            p *= np.exp(alpha * rnd / total_rounds)
        elif strategy == "exp_dec":
            alpha = np.log(p / (1 - p))
            p *= np.exp(-alpha * rnd / total_rounds)
        elif strategy == "log_inc":
            alpha = np.exp(1 / p) - 1
            p *= np.log2(1 + alpha * rnd / total_rounds)
        return p

    def _select_ids_based_on_algo(self, algo: str) -> List[int]:
        """
        Selects IDs based on the specified algorithm.
        """
        if algo == "random":
            topology = RandomTopology()
            return topology.get_selected_ids(self.node_id, self.config, self.reprs_dict, self.communities)
        if algo == "ring":
            topology = RingTopology()
            return topology.get_selected_ids(self.node_id, self.config)
        if algo == "grid":
            topology = GridTopology()
            return topology.get_selected_ids(self.node_id, self.config)
        if algo == "torus":
            topology = TorusTopology()
            return topology.get_selected_ids(self.node_id, self.config)
        return []

    def _apply_aggr_weight_strategy(self, weight: float, rnd: int, total_rounds: int) -> float:
        """
        Applies the aggregation weight strategy.
        """
        strategy = self.config.get("aggr_weight_strategy", None)
        if strategy is not None:
            init_weight = 0.1
            target_weight = 0.5
            if strategy == "linear":
                target_round = total_rounds // 2
                weight = 1 - (init_weight + (target_weight - init_weight) * (min(1, rnd / target_round)))
            elif strategy == "log":
                alpha = 0.05
                weight = 1 - (init_weight + (target_weight - init_weight) * (np.log(alpha * (rnd / total_rounds) + 1) / np.log(alpha + 1)))
            else:
                raise ValueError(f"Aggregation weight strategy {strategy} not implemented")
        return weight

    def _calculate_collab_weight(self, idx: int, own_aggr_weight: float, selected_ids: List[int]) -> float:
        """
        Calculates the collaborator weight.
        """
        if idx == self.node_id:
            return own_aggr_weight
        return (1 - own_aggr_weight) / (len(selected_ids) - 1)

    def get_representation(self) -> Dict[str, torch.Tensor]:
        """
        Returns the model weights as representation.
        """
        return self.get_model_weights()

    def mask_last_layer(self) -> None:
        """
        Masks the last layer of the model.
        """
        wts = self.get_model_weights()
        keys = self.model_utils.get_last_layer_keys(wts)
        key = [k for k in keys if "weight" in k][0]
        weight = torch.zeros_like(wts[key])
        weight[self.classes_of_interest] = wts[key][self.classes_of_interest]
        self.model.load_state_dict({key: weight.to(self.device)}, strict=False)

    def freeze_model_except_last_layer(self) -> None:
        """
        Freezes the model parameters except for the last layer.
        """
        wts = self.get_model_weights()
        keys = self.model_utils.get_last_layer_keys(wts)

        for name, param in self.model.named_parameters():
            if name not in keys:
                param.requires_grad = False

    def unfreeze_model(self) -> None:
        """
        Unfreezes all model parameters.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def flatten_repr(self, repr_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flattens the representation dictionary into a single tensor.
        """
        params = [repr_dict[key].view(-1) for key in repr_dict.keys()]
        return torch.cat(params)

    def compute_pseudo_grad_norm(self, prev_wts: Dict[str, torch.Tensor], new_wts: Dict[str, torch.Tensor]) -> float:
        """
        Computes the pseudo gradient norm.
        """
        return np.linalg.norm(self.flatten_repr(prev_wts) - self.flatten_repr(new_wts))

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError("Start round different from 0 not implemented yet")
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        for rnd in range(start_round, total_rounds):
            stats = {}

            # Wait on server to start the round
            self.comm_utils.receive(node_ids=self.server_node, tag=self.tag.ROUND_START)

            if self.config.get("finetune_last_layer", False):
                self.freeze_model_except_last_layer()

            # Train locally and send the representation to the server
            if not self.config.get("local_train_after_aggr", False):
                stats["train_loss"], stats["train_acc"] = self.local_train(epochs_per_round)

            repr_dict = self.get_representation()
            self.comm_utils.send(dest=self.server_node, data=repr_dict, tag=self.tag.REPR_ADVERT)

            # Collect the representations from all other nodes from the server
            reprs = self.comm_utils.receive(node_ids=self.server_node, tag=self.tag.REPRS_SHARE)
            reprs_dict = {k: v for k, v in enumerate(reprs, 1)}

            # Aggregate the representations based on the collaborator weights
            collab_weights_dict = self.get_collaborator_weights(reprs_dict, rnd)
            models_wts = reprs_dict

            layers_to_ignore = self.model_keys_to_ignore
            active_collab = {k for k, v in collab_weights_dict.items() if v > 0}
            inter_commu_last_layer_to_aggr = self.config.get("inter_commu_layer", None)
            if inter_commu_last_layer_to_aggr is not None and len(
                set(self.communities[self.node_id]).intersection(active_collab)
            ) != len(active_collab):
                layer_idx = self.model_utils.models_layers_idx[self.config["model"]][inter_commu_last_layer_to_aggr]
                layers_to_ignore = self.model_keys_to_ignore + list(list(models_wts.values())[0].keys())[layer_idx + 1:]

            avg_wts = self.weighted_aggregate(models_wts, collab_weights_dict, keys_to_ignore=layers_to_ignore)
            self.set_model_weights(avg_wts, layers_to_ignore)

            if self.config.get("train_only_fc", False):
                self.mask_last_layer()
                self.freeze_model_except_last_layer()
                self.local_train(1)
                self.unfreeze_model()

            stats["test_acc_before_training"] = self.local_test()

            # Train locally and send the representation to the server
            if self.config.get("local_train_after_aggr", False):
                prev_wts = self.get_model_weights()
                stats["train_loss"], stats["train_acc"] = self.local_train(epochs_per_round)
                new_wts = self.get_model_weights()
                stats["pseudo grad norm"] = self.compute_pseudo_grad_norm(prev_wts, new_wts)
                stats["test_acc_after_training"] = self.local_test()

            collab_weight = np.zeros(self.config["num_users"])
            for k, v in collab_weights_dict.items():
                collab_weight[k - 1] = v
            stats["Collaborator weights"] = collab_weight

            self.comm_utils.send(dest=self.server_node, data=stats, tag=self.tag.ROUND_STATS)


class FedStaticServer(BaseFedAvgServer):
    """
    Federated Static Server Class.
    """
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"

    def test(self) -> float:
        """
        Test the model on the server.
        """
        _, acc = self.model_utils.test(self.model, self._test_loader, self.loss_fn, self.device)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def single_round(self) -> List[Dict[str, Any]]:
        """
        Runs the whole training procedure for a single round.
        """
        for client_node in self.users:
            self.comm_utils.send(dest=client_node, data=None, tag=self.tag.ROUND_START)
        self.log_utils.log_console("Server waiting for all clients to finish local training")

        models = self.comm_utils.all_gather(self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models")

        self.send_representations(models)
        clients_round_stats = self.comm_utils.all_gather(self.tag.ROUND_STATS)
        self.log_utils.log_console("Server received all clients stats")

        self.log_utils.log_tb_round_stats(clients_round_stats, ["Collaborator weights"], self.round)

        self.log_utils.log_console(
            f"Round test acc before local training {[stats['test_acc_before_training'] for stats in clients_round_stats]}"
        )
        self.log_utils.log_console(
            f"Round test acc after local training {[stats['test_acc_after_training'] for stats in clients_round_stats]}"
        )

        return clients_round_stats

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the server.
        """
        self.log_utils.log_console("Starting static P2P collaboration")
        start_round = self.config.get("start_round", 0)
        total_round = self.config["rounds"]

        stats = []
        for rnd in range(start_round, total_round):
            self.round = rnd
            self.log_utils.log_console(f"Starting round {rnd}")
            round_stats = self.single_round()
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)
