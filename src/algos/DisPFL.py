"""
This module defines the DisPFLClient and DisPFLServer classes for distributed personalized federated learning.
"""

import copy
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from algos.base_class import BaseClient, BaseServer


class CommProtocol:
    """
    Communication protocol tags for the server and clients.
    """

    DONE: int = (
        0  # Used to signal the server that the client is done with local training
    )
    START: int = 1  # Used to signal by the server to start the current round
    UPDATES: int = 2  # Used to send the updates from the server to the clients
    LAST_ROUND: int = 3
    SHARE_MASKS: int = 4
    SHARE_WEIGHTS: int = 5
    FINISH: int = 6  # Used to signal the server to finish the current round


class DisPFLClient(BaseClient):
    """
    Client class for DisPFL (Distributed Personalized Federated Learning).

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with hyperparameters and paths.
        params (Optional[Dict[str, Tensor]]): Model parameters for local training.
        mask (Optional[OrderedDict[str, Tensor]]): Masks for model pruning.
        index (Optional[int]): The index of the client node.
        repr (Optional[OrderedDict[str, Tensor]]): Model representation (weights).
        dense_ratio (float): Ratio of dense layers in the model.
        anneal_factor (float): Annealing factor for model pruning.
        dis_gradient_check (bool): Whether to check the gradient during pruning.
        server_node (int): Index of the server node.
        num_users (int): Total number of users (clients) in the system.
        neighbors (List[int]): List of neighboring nodes for communication.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DisPFLClient class.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing settings and hyperparameters.
        """
        super().__init__(config)
        self.params: Optional[Dict[str, Tensor]] = None
        self.mask: Optional[OrderedDict[str, Tensor]] = None
        self.index: Optional[int] = None
        self.repr: Optional[OrderedDict[str, Tensor]] = None
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = (
            f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        )
        self.dense_ratio: float = self.config["dense_ratio"]
        self.anneal_factor: float = self.config["anneal_factor"]
        self.dis_gradient_check: bool = self.config["dis_gradient_check"]
        self.server_node: int = 1  # leader node
        self.num_users: int = config["num_users"]
        self.neighbors: List[int] = list(range(self.num_users))

        if self.node_id == 1:
            self.clients = list(range(2, self.num_users + 1))

    def local_train(self) -> None:
        """
        Train the model locally.
        """
        loss, acc = self.model_utils.train_mask(
            self.model, self.mask, self.optim, self.dloader, self.loss_fn, self.device
        )
        print(f"Node{self.node_id} train loss: {loss}, train acc: {acc}")

    def local_test(self, **kwargs: Any) -> Tuple[float, float]:
        """
        Test the model locally, not to be used in the traditional FedAvg.

        Args:
            kwargs (Any): Additional arguments for testing (not used).

        Returns:
            Tuple[float, float]: Test loss and accuracy.
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        return test_loss, acc

    def get_trainable_params(self) -> Dict[str, Tensor]:
        """
        Retrieves the trainable parameters of the model.

        Returns:
            Dict[str, Tensor]: A dictionary of model parameters.
        """
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Retrieve the current model weights.

        Returns:
            OrderedDict[str, Tensor]: The model weights.
        """
        return self.model.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]) -> None:
        """
        Set the model weights.

        Args:
            representation (OrderedDict[str, Tensor]): The model weights to be set.
        """
        self.model.load_state_dict(representation)

    def fire_mask(
        self, masks: OrderedDict[str, Tensor], round_num: int, total_round: int
    ) -> Tuple[OrderedDict[str, Tensor], Dict[str, int]]:
        """
        Fire mask method for model pruning.

        Args:
            masks (OrderedDict[str, Tensor]): The current mask.
            round_num (int): The current round number.
            total_round (int): Total number of rounds.

        Returns:
            Tuple[OrderedDict[str, Tensor], Dict[str, int]]: The updated masks and the number of elements removed.
        """
        weights = self.get_representation()
        drop_ratio = (
            self.anneal_factor / 2 * (1 + np.cos((round_num * np.pi) / total_round))
        )
        new_masks = copy.deepcopy(masks)
        num_remove: Dict[str, int] = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(
                masks[name] > 0,
                torch.abs(weights[name]),
                100000 * torch.ones_like(weights[name]),
            )
            idx = torch.sort(temp_weights.view(-1).to(self.device))
            new_masks[name].view(-1)[idx[1][: num_remove[name]]] = 0
        return new_masks, num_remove

    def regrow_mask(
        self,
        masks: OrderedDict[str, Tensor],
        num_remove: Dict[str, int],
        gradient: Optional[Dict[str, Tensor]] = None,
    ) -> OrderedDict[str, Tensor]:
        """
        Regrow mask method for model pruning.

        Args:
            masks (OrderedDict[str, Tensor]): The current mask.
            num_remove (Dict[str, int]): Number of elements removed from the mask.
            gradient (Optional[Dict[str, Tensor]]): Gradient information for regrowing the mask.

        Returns:
            OrderedDict[str, Tensor]: The updated mask after regrowth.
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
                
                del sort_temp
                
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

    def aggregate(
        self,
        nei_indexes: List[int],
        weights_lstrnd: List[OrderedDict[str, Tensor]],
        masks_lstrnd: List[OrderedDict[str, Tensor]],
    ) -> Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]:
        """
        Aggregate the model weights from neighboring clients.

        Args:
            nei_indexes (List[int]): Indices of neighboring clients.
            weights_lstrnd (List[OrderedDict[str, Tensor]]): List of weights from the last round.
            masks_lstrnd (List[OrderedDict[str, Tensor]]): List of masks from the last round.

        Returns:
            Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]: The aggregated model weights and masked weights.
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
                    w_tmp[k] += torch.from_numpy(count_mask[k]).to(
                        self.device
                    ) * weights_lstrnd[clnt][k].to(self.device)
                else:
                    w_tmp[k] = weights_lstrnd[self.index][k]
        w_p_g = copy.deepcopy(w_tmp)
        for name in self.mask:
            w_tmp[name] = w_tmp[name] * self.mask[name].to(self.device)
        return w_tmp, w_p_g

    def send_representations(self, representation: OrderedDict[str, Tensor]) -> None:
        """
        Send the model representation (weights) to other clients.

        Args:
            representation (OrderedDict[str, Tensor]): The model representation to be sent.
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node, representation, self.tag.UPDATES)
        print(f"Node 1 sent average weight to {len(self.clients)} nodes")

    def calculate_sparsities(
        self,
        params: Dict[str, Tensor],
        tabu: Optional[List[str]] = None,
        distribution: str = "ERK",
        sparse: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate sparsities for model pruning based on different distributions.

        Args:
            params (Dict[str, Tensor]): Model parameters to calculate sparsity for.
            tabu (Optional[List[str]]): List of parameters to exclude from pruning.
            distribution (str): The type of distribution to use for sparsity calculation.
            sparse (float): Sparsity ratio.

        Returns:
            Dict[str, float]: Sparsity values for each model parameter.
        """
        if tabu is None:
            tabu = []
        sparsities: Dict[str, float] = {}
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

    def init_masks(
        self, params: Dict[str, Tensor], sparsities: Dict[str, float]
    ) -> OrderedDict[str, Tensor]:
        """
        Initialize masks for model pruning based on sparsity values.

        Args:
            params (Dict[str, Tensor]): Model parameters to prune.
            sparsities (Dict[str, float]): Sparsity values for each parameter.

        Returns:
            OrderedDict[str, Tensor]: Initialized masks for the model parameters.
        """
        masks = OrderedDict()
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1 - sparsities[name]) * masks[name].numel())
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] = 1
        return masks

    def screen_gradient(self) -> Dict[str, Tensor]:
        """
        Screen the gradient of the model during training.

        Returns:
            Dict[str, Tensor]: The gradients of the model parameters.
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

    def hamming_distance(
        self, mask_a: OrderedDict[str, Tensor], mask_b: OrderedDict[str, Tensor]
    ) -> Tuple[int, int]:
        """
        Calculate the Hamming distance between two masks.

        Args:
            mask_a (OrderedDict[str, Tensor]): The first mask.
            mask_b (OrderedDict[str, Tensor]): The second mask.

        Returns:
            Tuple[int, int]: The number of differing elements and the total number of elements.
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
        round_idx: int,  # pylint: disable=unused-argument
        cur_clnt: int,
        client_num_in_total: int,
        client_num_per_round: int,
        dist_local: Optional[np.ndarray],  # pylint: disable=unused-argument
        total_dist: Optional[np.ndarray],  # pylint: disable=unused-argument
        cs: bool = False,
        active_ths_rnd: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Benefit choose method for client selection during federated learning.

        Args:
            round_idx (int): The current round index.
            cur_clnt (int): The current client index.
            client_num_in_total (int): The total number of clients.
            client_num_per_round (int): The number of clients per round.
            dist_local (Optional[np.ndarray]): Local distances (not used).
            total_dist (Optional[np.ndarray]): Total distances (not used).
            cs (bool): Whether to use client selection.
            active_ths_rnd (Optional[np.ndarray]): Thresholds for active clients.

        Returns:
            np.ndarray: Array of selected client indexes.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = np.array(list(range(client_num_in_total)))
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

    def model_difference(
        self, model_a: OrderedDict[str, Tensor], model_b: OrderedDict[str, Tensor]
    ) -> Tensor:
        """
        Calculate the difference between two models.

        Args:
            model_a (OrderedDict[str, Tensor]): The first model.
            model_b (OrderedDict[str, Tensor]): The second model.

        Returns:
            Tensor: The difference between the two models.
        """
        diff = sum(
            [torch.sum(torch.square(model_a[name] - model_b[name])) for name in model_a]
        )
        return diff

    def run_protocol(self) -> None:
        """
        Runs the entire training protocol for federated learning.
        """
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        self.params = self.get_trainable_params()
        sparsities = self.calculate_sparsities(
            self.params, sparse=self.dense_ratio
        )  # calculate sparsity to create masks
        self.mask = self.init_masks(self.params, sparsities)  # mask_per_local
        dist_locals = np.zeros(shape=self.num_users)
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
            print(f"Node {self.index}'s neighbors index:{[i + 1 for i in nei_indexes]}")

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
            acc = self.local_test()
            print(f"Node {self.node_id} local test: {acc[1]}")
            representation = self.get_representation()
            if not self.config["static"]:
                if not self.dis_gradient_check:
                    gradient = self.screen_gradient()
                self.mask, num_remove = self.fire_mask(self.mask, epoch, total_epochs)
                self.mask = self.regrow_mask(self.mask, num_remove, gradient)
            self.comm_utils.send_signal(
                dest=0, data=copy.deepcopy(representation), tag=self.tag.SHARE_WEIGHTS
            )

            # test updated model
            self.set_representation(representation)
            acc = self.local_test()
            self.comm_utils.send_signal(dest=0, data=acc[1], tag=self.tag.FINISH)


class DisPFLServer(BaseServer):
    """
    Server class for DisPFL (Distributed Personalized Federated Learning).

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with hyperparameters and paths.
        best_acc (float): The best accuracy achieved so far.
        round (int): Current training round.
        masks (Any): Masks received from clients.
        reprs (Any): Representations (weights) received from clients.
        dense_ratio (float): Ratio of dense layers in the model.
        num_users (int): Total number of users (clients) in the system.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DisPFLServer class.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing settings and hyperparameters.
        """
        super().__init__(config)
        self.best_acc: float = 0
        self.round: int = 0
        self.masks: Any = 0
        self.reprs: Any = 0
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = (
            f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        )
        self.dense_ratio: float = self.config["dense_ratio"]
        self.num_users: int = self.config["num_users"]

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Retrieve the current model weights.

        Returns:
            OrderedDict[str, Tensor]: The model weights.
        """
        return self.model.state_dict()

    def send_representations(
        self, representations: Dict[int, OrderedDict[str, Tensor]]
    ) -> None:
        """
        Send model representations (weights) to the clients.

        Args:
            representations (Dict[int, OrderedDict[str, Tensor]]): The model representations to be sent to clients.
        """
        for client_node in self.users:
            self.comm_utils.send_signal(client_node, representations, self.tag.UPDATES)
            self.log_utils.log_console(
                f"Server sent {len(representations)} representations to node {client_node}"
            )

    def test(self) -> float:
        """
        Test the model on the server.

        Returns:
            float: The accuracy of the model on the test set.
        """
        acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc[1] > self.best_acc:
            self.best_acc = acc[1]
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc[1]

    def single_round(self, epoch: int, active_ths_rnd: np.ndarray) -> None:
        """
        Executes a single round of training and communication with clients.

        Args:
            epoch (int): The current epoch (round number).
            active_ths_rnd (np.ndarray): Array indicating which clients are active in the current round.
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

    def get_trainable_params(self) -> Dict[str, Tensor]:
        """
        Retrieve the trainable parameters of the model.

        Returns:
            Dict[str, Tensor]: A dictionary of model parameters.
        """
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = param
        return param_dict

    def run_protocol(self) -> None:
        """
        Runs the entire training protocol for federated learning.
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
