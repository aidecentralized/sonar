"""
This module defines the DefKTClient and DefKTServer classes for federated learning using a knowledge
transfer approach.
"""

import copy
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from utils.communication.comm_utils import CommunicationManager
import torch.nn as nn

from algos.base_class import BaseClient, BaseServer


class CommProtocol:
    """
    Communication protocol tags for the server and clients
    """

    DONE: int = (
        0  # Used to signal the server that the client is done with local training
    )
    START: int = 1  # Used to signal by the server to start the current round
    UPDATES: int = 2  # Used to send the updates from the server to the clients
    FINISH: int = 3  # Used to signal the server to finish the current round


class DefKTClient(BaseClient):
    """
    Client class for DefKT (Deep Mutual Learning with Knowledge Transfer)
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = (
            f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        )
        self.server_node = 1  # leader node
        self.best_acc = 0.0  # Initialize best accuracy attribute
        if self.node_id == 1:
            self.num_users = config["num_users"]
            self.clients = list(range(2, self.num_users + 1))

    def local_train(self) -> None:
        """
        Train the model locally
        """
        avg_loss = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device
        )

    def local_test(self, **kwargs: Any) -> float:
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def deep_mutual_train(self, teacher_repr: OrderedDict[str, Tensor]) -> None:
        """
        Train the model locally with deep mutual learning
        """
        teacher_model = copy.deepcopy(self.model)
        teacher_model.load_state_dict(teacher_repr)
        print(f"Deep mutual learning at student Node {self.node_id}")
        avg_loss, acc = self.model_utils.deep_mutual_train(
            [self.model, teacher_model], self.optim, self.dloader, self.device
        )

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]) -> None:
        """
        Set the model weights
        """
        self.model.load_state_dict(representation)

    def fed_avg(
        self, model_wts: List[OrderedDict[str, Tensor]]
    ) -> OrderedDict[str, Tensor]:
        """
        Federated averaging of model weights
        """
        num_users = len(model_wts)
        coeff = 1 / num_users
        avgd_wts = OrderedDict()
        first_model = model_wts[0]

        for client_num in range(num_users):
            local_wts = model_wts[client_num]
            for key in first_model.keys():
                if client_num == 0:
                    avgd_wts[key] = coeff * local_wts[key].to(self.device)
                else:
                    avgd_wts[key] += coeff * local_wts[key].to(self.device)
        return avgd_wts

    def aggregate(
        self, representation_list: List[OrderedDict[str, Tensor]]
    ) -> OrderedDict[str, Tensor]:
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def send_representations(self, representation: OrderedDict[str, Tensor]) -> None:
        """
        Send the model representations to the clients
        """
        for client_node in self.clients:
            self.comm_utils.send(client_node, representation, tag=self.tag.UPDATES)
        print(f"Node 1 sent average weight to {len(self.clients)} nodes")

    def single_round(
        self, self_repr: OrderedDict[str, Tensor]
    ) -> OrderedDict[str, Tensor]:
        """
        Runs a single training round
        """
        print("Node 1 waiting for all clients to finish")
        reprs = self.comm_utils.all_gather(tag=self.tag.DONE)
        reprs.append(self_repr)
        print(f"Node 1 received {len(reprs)} clients' weights")
        avg_wts = self.aggregate(reprs)
        self.send_representations(avg_wts)
        return avg_wts

    def assign_own_status(self, status: List[List[int]]) -> None:
        """
        Assign the status (teacher/student) to the client
        """
        if self.node_id in status[0]:
            self.status = "teacher"
            index = status[0].index(self.node_id)
            self.pair_id = status[1][index]
        elif self.node_id in status[1]:
            self.status = "student"
            index = status[1].index(self.node_id)
            self.pair_id = status[0][index]
        else:
            self.status = None
            self.pair_id = None
        print(f"Node {self.node_id} is a {self.status}, pair with {self.pair_id}")

    def run_protocol(self) -> None:
        """
        Runs the entire training protocol
        """
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for epoch in range(start_epochs, total_epochs):
            status = self.comm_utils.receive(0, tag=self.tag.START)
            self.assign_own_status(status)
            if self.status == "teacher":
                self.local_train()
                self_repr = self.get_representation()
                self.comm_utils.send(
                    dest=self.pair_id, data=self_repr, tag=self.tag.DONE
                )
                print(f"Node {self.node_id} sent repr to student node {self.pair_id}")
            elif self.status == "student":
                teacher_repr = self.comm_utils.receive(self.pair_id, tag=self.tag.DONE)
                print(
                    f"Node {self.node_id} received repr from teacher node {self.pair_id}"
                )
                self.deep_mutual_train(teacher_repr)
            else:
                print(f"Node {self.node_id} do nothing")
            acc = self.local_test()
            print(f"Node {self.node_id} test_acc:{acc:.4f}")
            self.comm_utils.send(0, data=acc, tag=self.tag.FINISH)


class DefKTServer(BaseServer):
    """
    Server class for DefKT (Deep Mutual Learning with Knowledge Transfer)
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = (
            f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        )
        self.best_acc = 0.0  # Initialize best accuracy attribute

    def send_representations(
        self, representations: Dict[int, OrderedDict[str, Tensor]]
    ) -> None:
        """
        Send the model representations to the clients
        """
        for client_node in self.users:
            self.comm_utils.send(client_node, representations, self.tag.UPDATES)
            self.log_utils.log_console(
                f"Server sent {len(representations)} representations to node {client_node}"
            )

    def test(self) -> float:
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def assigns_clients(self) -> Optional[Tuple[List[int], List[int]]]:
        """
        Assigns clients as teachers and students
        """
        num_teachers = self.config["num_teachers"]
        clients = list(range(1, self.num_users + 1))
        if 2 * num_teachers > self.num_users:
            return None  # Not enough room to pick two non-overlapping subarrays
        selected_indices = random.sample(range(self.num_users), 2 * num_teachers)
        selected_elements = [clients[i] for i in selected_indices]
        teachers = selected_elements[:num_teachers]
        students = selected_elements[num_teachers:]
        return teachers, students

    def single_round(self) -> None:
        """
        Runs a single training round
        """
        teachers, students = self.assigns_clients()  # type: ignore
        self.log_utils.log_console(f"Teachers:{teachers}")
        self.log_utils.log_console(f"Students:{students}")
        for client_node in self.users:
            self.log_utils.log_console(
                f"Server sending status from {self.node_id} to {client_node}"
            )
            self.comm_utils.send(
                dest=client_node, data=[teachers, students], tag=self.tag.START
            )

    def run_protocol(self) -> None:
        """
        Runs the entire training protocol
        """
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for epoch in range(start_epochs, total_epochs):
            self.log_utils.log_console(f"Starting round {epoch}")
            self.single_round()
            accs = self.comm_utils.all_gather(tag=self.tag.FINISH)
            self.log_utils.log_console(f"Round {epoch} done; acc {accs}")
