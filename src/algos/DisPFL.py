"""
This module defines the DisPFLClient and DisPFLServer classes for distributed personalized federated learning.
"""

import copy
import math
import random
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, from_numpy, numel, randperm, zeros_like

from algos.base_class import BaseClient, BaseServer


class CommProtocol:
    """
    Communication protocol tags for the server and clients.
    """

    DONE = 0  # Used to signal the server that the client is done with local training
    START = 1  # Used to signal by the server to start the current round
    UPDATES = 2  # Used to send the updates from the server to the clients
    LAST_ROUND = 3
    SHARE_MASKS = 4
    SHARE_WEIGHTS = 5
    FINISH = 6  # Used to signal the server to finish the current round


class DisPFLClient(BaseClient):
    """
    Client class for DisPFL (Distributed Personalized Federated Learning).
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        self.dense_ratio = self.config["dense_ratio"]
        self.anneal_factor = self.config["anneal_factor"]
        self.dis_gradient_check = self.config["dis_gradient_check"]
        self.server_node = 1  # leader node
        self.num_users = config["num_users"]
        self.neighbors = list(range(self.num_users))
        if self.node_id == 1:
            self.clients = list(range(2, self.num_users + 1))

    def local_train(self):
        """
        Train the model locally.
        """
        loss, acc = self.model_utils.train_mask(
            self.model, self.mask, self.optim, self.dloader, self.loss_fn, self.device
        )
        print(f"Node{self.node_id} train loss: {loss}, train acc: {acc}")

    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg.
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy so far
        # if acc > self.best_acc:
        #     self.best_acc = acc
        #     self.model_utils.save_model(self.model, self.model_save_path)
        return test_loss, acc

    def get_trainable_params(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights.
        """
        return self.model.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model weights.
        """
        self.model.load_state_dict(representation)

    def fire_mask(self, masks, round_num, total_round):
        """
        Fire mask method for model pruning.
        """
        weights = self.get_representation()
        drop_ratio = (
            self.anneal_factor / 2 * (1 + np.cos((round_num * np.pi) / total_round))
        )
        new_masks = copy.deepcopy(masks)
        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(
                masks[name] > 0,
                torch.abs(weights[name]),
                100000 * torch.ones_like(weights[name]),
            )
            x, idx = torch.sort(temp_weights.view(-1).to(self.device))
            new_masks[name].view(-1)[idx[: num_remove[name]]] = 0
        return new_masks, num_remove

    def regrow_mask(self, masks, num_remove, gradient=None):
        """
        Regrow mask method for model pruning.
        """
        new_masks = copy.deepcopy(masks)
        for name in masks:
            if not self.dis_gradient_check:
                temp = torch.where(
                    masks[name] == 0,
                    torch.abs(gradient[name]).to(self.device),
                    -100000 * torch.ones_like(gradient[name]).to(self.device),
                )
                sort_temp, idx = torch.sort(
                    temp.view(-1).to(self.device), descending=True
                )
                new_masks[name].view(-1)[idx[: num_remove[name]]] = 1
            else:
                temp = torch.where(
                    masks[name] == 0,
                    torch.ones_like(masks[name]),
                    torch.zeros_like(masks[name]),
                )
                idx = torch.multinomial(
                    temp.flatten().to(self.device), num_remove[name], replacement=False
                )
                new_masks[name].view(-1)[idx] = 1
        return new_masks

    def aggregate(self, nei_indexes, weights_lstrnd, masks_lstrnd):
        """
        Aggregate the model weights.
        """
        count_mask = copy.deepcopy(masks_lstrnd[self.index])
        for k in count_mask.keys():
            count_mask[k] = count_mask[k] - count_mask[k]  # zero out by pruning
            for clnt in nei_indexes:
                count_mask[k] += masks_lstrnd[clnt][k].to(self.device)  # mask
        for k in count_mask.keys():
            count_mask[k] = np.divide(
                1,
                count_mask[k].cpu(),
                out=np.zeros_like(count_mask[k].cpu()),
                where=count_mask[k].cpu() != 0,
            )

        # update weight temp
        w_tmp = copy.deepcopy(weights_lstrnd[self.index])
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in self.neighbors:
                if k in self.params:
                    w_tmp[k] += from_numpy(count_mask[k]).to(
                        self.device
                    ) * weights_lstrnd[clnt][k].to(self.device)
                else:
                    w_tmp[k] = weights_lstrnd[self.index][k]
        w_p_g = copy.deepcopy(w_tmp)
        for name in self.mask:
            w_tmp[name] = w_tmp[name] * self.mask[name].to(self.device)
        return w_tmp, w_p_g

    def send_representations(self, representation):
        """
        Set the model.
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node, representation, self.tag.UPDATES)
        print(f"Node 1 sent average weight to {len(self.clients)} nodes")

    def calculate_sparsities(self, params, tabu=None, distribution="ERK", sparse=0.5):
        """
        Calculate sparsities for model pruning.
        """
        if tabu is None:
            tabu = []
        sparsities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    sparsities[name] = 1 - self.dense_ratio
                else:
                    sparsities[name] = 0
        elif distribution == "ERK":
            print("initialize by ERK")
            total_params = 0
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            dense_layers = set()
            density = sparse
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                            np.sum(params[name].shape) / np.prod(params[name].shape)
                        ) ** self.config["erk_power_scale"]
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaining layers.
            for name in params:
                if name in dense_layers:
                    sparsities[name] = 0
                else:
                    sparsities[name] = 1 - epsilon * raw_probabilities[name]
        return sparsities

    def init_masks(self, params, sparsities):
        """
        Initialize masks for model pruning.
        """
        masks = OrderedDict()
        for name in params:
            masks[name] = zeros_like(params[name])
            dense_numel = int((1 - sparsities[name]) * numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] = 1
        return masks

    def screen_gradient(self):
        """
        Screen gradient method for model pruning.
        """
        model = self.model
        model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)
        model.zero_grad()
        (x, labels) = next(iter(self.dloader))
        x, labels = x.to(self.device), labels.to(self.device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient = {}
        for name, param in self.model.named_parameters():
            gradient[name] = param.grad.to("cpu")

        return gradient

    def hamming_distance(self, mask_a, mask_b):
        """
        Calculate the Hamming distance between two masks.
        """
        dis = 0
        total = 0
        for key in mask_a:
            dis += torch.sum(
                mask_a[key].int().to(self.device) ^ mask_b[key].int().to(self.device)
            )
            total += mask_a[key].numel()
        return dis, total

    def _benefit_choose(
        self,
        round_idx,
        cur_clnt,
        client_num_in_total,
        client_num_per_round,
        dist_local,
        total_dist,
        cs=False,
        active_ths_rnd=None,
    ):
        """
        Benefit choose method for client selection.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = list(range(client_num_in_total))
            return client_indexes

        if cs == "random":
            num_users = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(
                range(client_num_in_total), num_users, replace=False
            )
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(
                    range(client_num_in_total), num_users, replace=False
                )

        elif cs == "ring":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            client_indexes = np.array(np.where(active_ths_rnd == 1)).squeeze()
            client_indexes = np.delete(
                client_indexes, int(np.where(client_indexes == cur_clnt)[0])
            )
        return client_indexes

    def model_difference(self, model_a, model_b):
        """
        Calculate the difference between two models.
        """
        diff = sum(
            [torch.sum(torch.square(model_a[name] - model_b[name])) for name in model_a]
        )
        return diff

    def run_protocol(self):
        """
        Runs the entire training protocol.
        """
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        self.params = self.get_trainable_params()
        sparsities = self.calculate_sparsities(
            self.params, sparse=self.dense_ratio
        )  # calculate sparsity to create masks
        self.mask = self.init_masks(self.params, sparsities)  # mask_per_local
        dist_locals = np.zeros(shape=(self.num_users))
        self.index = self.node_id - 1
        masks_lstrnd = [self.mask for i in range(self.num_users)]
        weights_lstrnd = [
            copy.deepcopy(self.get_representation()) for i in range(self.num_users)
        ]
        w_per_globals = [
            copy.deepcopy(self.get_representation()) for i in range(self.num_users)
        ]
        for epoch in range(start_epochs, total_epochs):
            # wait for signal to start round
            active_ths_rnd = self.comm_utils.wait_for_signal(src=0, tag=self.tag.START)
            if epoch != 0:
                [weights_lstrnd, masks_lstrnd] = self.comm_utils.wait_for_signal(
                    src=0, tag=self.tag.LAST_ROUND
                )
            self.repr = self.get_representation()
            dist_locals[self.index], total_dis = self.hamming_distance(
                masks_lstrnd[self.index], self.mask
            )
            print(
                f"Node{self.node_id}: local mask change {dist_locals[self.index]}/{total_dis}"
            )
            if active_ths_rnd[self.index] == 0:
                nei_indexes = np.array([])
            else:
                nei_indexes = self._benefit_choose(
                    epoch,
                    self.index,
                    self.num_users,
                    self.config["neighbors"],
                    dist_locals,
                    total_dis,
                    self.config["cs"],
                    active_ths_rnd,
                )
            if self.num_users != self.config["neighbors"]:
                nei_indexes = np.append(nei_indexes, self.index)
            print(
                f"Node {self.index}'s neighbors index:{[i + 1 for i in nei_indexes]}"
            )

            for tmp_idx in nei_indexes:
                if tmp_idx != self.index:
                    dist_locals[tmp_idx], _ = self.hamming_distance(
                        self.mask, masks_lstrnd[tmp_idx]
                    )

            if self.config["cs"] != "full":
                print(
                    f"choose client_indexes: {str(nei_indexes)}, according to {self.config['cs']}"
                )
            else:
                print(
                    f"choose client_indexes: {str(nei_indexes)}, according to {self.config['cs']}"
                )
            if active_ths_rnd[self.index] != 0:
                nei_distances = [dist_locals[i] for i in nei_indexes]
                print("choose mask diff: " + str(nei_distances))

            if active_ths_rnd[self.index] == 1:
                new_repr, w_per_globals[self.index] = self.aggregate(
                    nei_indexes, weights_lstrnd, masks_lstrnd
                )
            else:
                new_repr = copy.deepcopy(weights_lstrnd[self.index])
                w_per_globals[self.index] = copy.deepcopy(weights_lstrnd[self.index])
            model_diff = self.model_difference(new_repr, self.repr)
            print(f"Node {self.node_id} model_diff{model_diff}")
            self.comm_utils.send_signal(
                dest=0, data=copy.deepcopy(self.mask), tag=self.tag.SHARE_MASKS
            )

            self.set_representation(new_repr)

            # locally train
            print(f"Node {self.node_id} local train")
            self.local_train()
            loss, acc = self.local_test()
            print(f"Node {self.node_id} local test: {acc}")
            repr = self.get_representation()
            if not self.config["static"]:
                if not self.dis_gradient_check:
                    gradient = self.screen_gradient()
                self.mask, num_remove = self.fire_mask(self.mask, epoch, total_epochs)
                self.mask = self.regrow_mask(self.mask, num_remove, gradient)
            self.comm_utils.send_signal(
                dest=0, data=copy.deepcopy(repr), tag=self.tag.SHARE_WEIGHTS
            )

            # test updated model
            self.set_representation(repr)
            loss, acc = self.local_test()
            self.comm_utils.send_signal(dest=0, data=acc, tag=self.tag.FINISH)


class DisPFLServer(BaseServer):
    """
    Server class for DisPFL (Distributed Personalized Federated Learning).
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        self.dense_ratio = self.config["dense_ratio"]
        self.num_users = self.config["num_users"]

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights.
        """
        return self.model.state_dict()

    def send_representations(self, representations):
        """
        Set the model.
        """
        for client_node in self.users:
            self.comm_utils.send_signal(client_node, representations, self.tag.UPDATES)
            self.log_utils.log_console(
                f"Server sent {len(representations)} representations to node {client_node}"
            )

    def test(self) -> float:
        """
        Test the model on the server.
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def single_round(self, epoch, active_ths_rnd):
        """
        Runs the whole training procedure.
        """
        for client_node in self.users:
            self.log_utils.log_console(
                f"Server sending semaphore from {self.node_id} to {client_node}"
            )
            self.comm_utils.send_signal(
                dest=client_node, data=active_ths_rnd, tag=self.tag.START
            )
            if epoch != 0:
                self.comm_utils.send_signal(
                    dest=client_node,
                    data=[self.reprs, self.masks],
                    tag=self.tag.LAST_ROUND,
                )

        self.masks = self.comm_utils.wait_for_all_clients(
            self.users, self.tag.SHARE_MASKS
        )
        self.reprs = self.comm_utils.wait_for_all_clients(
            self.users, self.tag.SHARE_WEIGHTS
        )

    def get_trainable_params(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def run_protocol(self):
        """
        Runs the entire training protocol.
        """
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]

        for epoch in range(start_epochs, total_epochs):
            self.round = epoch
            active_ths_rnd = np.random.choice(
                [0, 1],
                size=self.num_users,
                p=[1.0 - self.config["active_rate"], self.config["active_rate"]],
            )
            self.log_utils.log_console(f"Starting round {epoch}")

            self.single_round(epoch, active_ths_rnd)

            accs = self.comm_utils.wait_for_all_clients(self.users, self.tag.FINISH)
            self.log_utils.log_console(f"Round {epoch} done; acc {accs}")
