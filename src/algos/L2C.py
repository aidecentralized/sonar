"""Module docstring: This module implements the L2C algorithm for federated learning."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, Tensor
from torch.nn import Module, Linear, ReLU

from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from algos.base_class import BaseFedAvgClient, BaseFedAvgServer


class L2CClient(BaseFedAvgClient):
    """Class docstring: This class represents a client in the L2C federated learning algorithm."""
    def __init__(self, config) -> None:
        super().__init__(config)

        self.init_collab_weights()
        self.sharing_mode = self.config["sharing"]

    def init_collab_weights(self):
        n = self.config["num_users"]
        # Neighbors id = [1, ..., num_users]
        # Neighbors idx = [0, ..., num_users - 1]
        self.neighbors_id_to_idx = {idx + 1: idx for idx in range(n)}

        # TODO Init not specified in the paper
        self.alpha = torch.ones(n, requires_grad=True)
        self.collab_weights = F.softmax(self.alpha, dim=0)
        self.alpha_optim = optim.Adam(
            [self.alpha],
            lr=self.config["alpha_lr"],
            weight_decay=self.config["alpha_weight_decay"],
        )
        # self.alpha_optim = optim.SGD([self.alpha], lr=self.config["alpha_lr"])

    def filter_out_worse_neighbors(self, num_neighbors_to_keep):
        """
        Keep only the top k neighbors (+ itself)
        """
        if num_neighbors_to_keep >= self.alpha.shape[0]:
            raise ValueError(
                "Number of neighbors to keep is greater than the number of neighbors"
            )

        # Only consider itself as active neighbors
        if num_neighbors_to_keep <= 0:
            self.neighbors_id_to_idx = {self.node_id: 0}
            self.alpha = torch.ones(1, requires_grad=True)
        else:
            # Select the top k neighbors (without itself)
            own_idx = self.neighbors_id_to_idx[self.node_id]
            own_mask = torch.ones(self.alpha.shape, dtype=torch.bool)
            own_mask[own_idx] = 0
            remaining_neighbors_idx = torch.topk(
                self.alpha[own_mask], k=num_neighbors_to_keep, largest=True
            )[1].tolist()
            # Fix the shift created by masking own id
            remaining_neighbors_idx = [
                idx + 1 if idx >= own_idx else idx for idx in remaining_neighbors_idx
            ]
            remaining_neighbors_idx.append(own_idx)

            # Get a list of id and idx of neighbors to keep
            remaining_neighbors_id_idx = [
                (id, idx)
                for id, idx in self.neighbors_id_to_idx.items()
                if idx in remaining_neighbors_idx
            ]

            # Sort the list by idx (as in alpha)
            remaining_neighbors_id_idx = sorted(
                remaining_neighbors_id_idx, key=lambda x: x[1]
            )
            sorted_idx = [idx for _, idx in remaining_neighbors_id_idx]
            self.alpha = self.alpha[sorted_idx].detach().requires_grad_(True)
            # Assign new idx to neighbors corresponding to their idx in alpha
            self.neighbors_id_to_idx = {
                id: new_idx
                for new_idx, (id, _) in enumerate(remaining_neighbors_id_idx)
            }

        self.collab_weights = F.softmax(self.alpha, dim=0)

        # TODO Not sure reint opti is the best solution
        self.alpha_optim = optim.Adam(
            [self.alpha],
            lr=self.config["alpha_lr"],
            weight_decay=self.config["alpha_weight_decay"],
        )
        # self.alpha_optim = optim.SGD([self.alpha], lr=self.config["alpha_lr"])

    def learn_collab_weights(self, models_update_wts):
        self.model.eval()
        alpha_loss, correct = 0, 0
        for data, target in self.val_dloader:
            data, target = data.to(self.device), target.to(self.device)

            self.alpha_optim.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)

            # Compute grad for weight of aggregated model
            loss.backward()

            # Create dict params(layer) name -> gradiant
            grad_dict = {k: v.grad for k, v in self.model.named_parameters()}

            # Compute grad for each alpha
            # Multiply weights and their corresponding grads and sum them by
            # corresponding collaborator weights
            collab_weights_grads = []
            for id in self.neighbors_id_to_idx.keys():
                cw_grad = 0
                for key in grad_dict.keys():
                    if key not in self.model_keys_to_ignore:
                        if self.sharing_mode == "updates":
                            cw_grad -= (
                                models_update_wts[id][key] *
                                grad_dict[key].cpu()
                            ).sum()
                        elif self.sharing_mode == "weights":
                            cw_grad += (
                                models_update_wts[id][key] *
                                grad_dict[key].cpu()
                            ).sum()
                        else:
                            raise ValueError("Unknown sharing mode")
                collab_weights_grads.append(cw_grad)

            # if self.node_id == 1:
            #     print(collab_weights_grad)
            #     print("Node {}'s weights before:{}, grad_fn {} ,grad {}".format(self.node_id, self.collab_weights, self.collab_weights.grad_fn, self.collab_weights.grad))

            self.collab_weights.backward(torch.tensor(collab_weights_grads))

            self.alpha_optim.step()

            alpha_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are
            # the same
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.train()
        acc = correct / len(self.val_dloader.dataset)
        print(
            "Node {}'s updating alpha loss:{}, acc: {}".format(
                self.node_id, alpha_loss, acc
            )
        )

        # Update collab weights based on alpha
        self.collab_weights = F.softmax(self.alpha, dim=0)

        return alpha_loss, acc

    def print_GPU_memory(self):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved

        print(
            f"Client {self.node_id} :GPU memory: reserved {r}, allocated {a}, free {f}"
        )

    def get_collaborator_weights(self, reprs_dict):
        """
        Returns the weights of the collaborators for the current round
        """
        # Make sure the specified index is within the range of the list
        indices = list(reprs_dict.keys())
        if self.node_id not in indices:
            raise ValueError("Own model not included in representations")

        collab_weights_dict = defaultdict(lambda: 0.0)
        for id, idx in self.neighbors_id_to_idx.items():
            collab_weights_dict[id] = self.collab_weights[idx]

        return collab_weights_dict

    def get_model_weights_without_keys_to_ignore(self):
        return self.model_utils.filter_model_weights(
            self.get_model_weights(), self.model_keys_to_ignore
        )

    def get_representation(self):
        if self.sharing_mode == "updates":

            return self.model_utils.substract_model_weights(
                self.prev_model, self.get_model_weights_without_keys_to_ignore()
            )
        elif self.sharing_mode == "weights":
            return self.get_model_weights()
        else:
            raise ValueError("Unknown sharing mode")

    def run_protocol(self):
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        for round in range(start_round, total_rounds):

            if self.sharing_mode == "updates":
                self.prev_model = self.get_model_weights_without_keys_to_ignore()

            # Collab weights compted in previous round are used in current
            # round
            cw = np.zeros(self.config["num_users"])
            for id, idx in self.neighbors_id_to_idx.items():
                cw[id - 1] = self.collab_weights[idx]
            round_stats = {
                "collab_weights": cw,
            }

            # Wait on server to start the round
            self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.ROUND_START
            )

            # Train locally and send the representation to the server
            round_stats["train_loss"], round_stats["train_acc"] = self.local_train(
                epochs_per_round
            )
            repr = self.get_representation()
            self.comm_utils.send_signal(
                dest=self.server_node, data=repr, tag=self.tag.REPR_ADVERT
            )

            # Collect the representations from all other nodes from the server
            reprs = self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.REPRS_SHARE
            )
            # In the future this dict might be generated by the server to send
            # only requested models
            reprs_dict = {k: v for k, v in enumerate(reprs, 1)}

            # Aggregate the representations based on the collab wheigts
            collab_weights_dict = self.get_collaborator_weights(reprs_dict)

            # Since clients representations are also used to transmit knowledge
            # There is no need to fetch the server for the selected clients'
            # knowledge
            models_update_wts = reprs_dict

            new_wts = self.weighted_aggregate(
                models_update_wts, collab_weights_dict, self.model_keys_to_ignore
            )

            if self.sharing_mode == "updates":
                new_wts = self.model_utils.substract_model_weights(
                    self.prev_model, new_wts
                )

            self.set_model_weights(new_wts, self.model_keys_to_ignore)

            # Test updated model
            round_stats["test_acc"] = self.local_test()

            round_stats["validation_loss"], round_stats["validation_acc"] = (
                self.learn_collab_weights(models_update_wts)
            )
            print("node {} weight: {}".format(
                self.node_id, self.collab_weights))

            # Lower the number of neighbors
            if round == self.config["T_0"]:
                self.filter_out_worse_neighbors(
                    self.config["target_clients_after_T_0"])

            self.comm_utils.send_signal(
                dest=self.server_node, data=round_stats, tag=self.tag.ROUND_STATS
            )


class L2CServer(BaseFedAvgServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def test(self) -> float:
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy so far
        # if acc > self.best_acc:
        #     self.best_acc = acc
        #     self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def single_round(self):
        """
        Runs the whole training procedure
        """

        # Send signal to all clients to start local training
        for client_node in self.users:
            self.comm_utils.send_signal(
                dest=client_node, data=None, tag=self.tag.ROUND_START
            )
        self.log_utils.log_console(
            "Server waiting for all clients to finish local training"
        )

        # Collect representations (from all clients
        reprs = self.comm_utils.wait_for_all_clients(
            self.users, self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models")

        # Broadcast the representations to all clients
        self.send_representations(reprs)

        # Collect round stats from all clients
        round_stats = self.comm_utils.wait_for_all_clients(
            self.users, self.tag.ROUND_STATS
        )
        self.log_utils.log_console("Server received all clients stats")

        # Log the round stats on tensorboard
        self.log_utils.log_tb_round_stats(
            round_stats, ["collab_weights"], self.round)

        self.log_utils.log_console(
            f"Round test acc {[stats['test_acc'] for stats in round_stats]}"
        )

        return round_stats

    def run_protocol(self):
        self.log_utils.log_console("Starting  L2C")
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]

        stats = []
        for round in range(start_round, total_rounds):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))

            round_stats = self.single_round()
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(
            stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)
