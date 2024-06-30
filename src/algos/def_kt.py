from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor, cat
import copy
import torch.nn as nn
import random

from algos.base_class import BaseClient, BaseServer


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """

    DONE = 0  # Used to signal the server that the client is done with local training
    START = 1  # Used to signal by the server to start the current round
    UPDATES = 2  # Used to send the updates from the server to the clients
    FINISH = 3  # Used to signal the server to finish the current round


class DefKTClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )
        self.server_node = 1  # leader node
        if self.node_id == 1:
            self.num_clients = config["num_clients"]
            self.clients = list(range(2, self.num_clients + 1))

    def local_train(self):
        """
        Train the model locally
        """
        avg_loss = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device
        )
        # print("Client {} finished training with loss {}".format(self.node_id, avg_loss))
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)

    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy
        # so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def deep_mutual_train(self, teacher_repr):
        """
        Train the model locally
        """
        teacher_model = copy.deepcopy(self.model)
        teacher_model.load_state_dict(teacher_repr)
        print("Deep mutual learning at student Node {}".format(self.node_id))
        avg_loss, acc = self.model_utils.deep_mutual_train(
            [self.model, teacher_model], self.optim, self.dloader, self.device
        )

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model weights
        """
        self.model.load_state_dict(representation)

    def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]):
        # All models are sampled currently at every round
        # Each model is assumed to have equal amount of data and hence
        # coeff is same for everyone
        num_clients = len(model_wts)
        coeff = 1 / num_clients
        avgd_wts = OrderedDict()
        first_model = model_wts[0]

        for client_num in range(num_clients):
            local_wts = model_wts[client_num]
            for key in first_model.keys():
                if client_num == 0:
                    avgd_wts[key] = coeff * local_wts[key].to(self.device)
                else:
                    avgd_wts[key] += coeff * local_wts[key].to(self.device)
        return avgd_wts

    def aggregate(self, representation_list: List[OrderedDict[str, Tensor]]):
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def send_representations(self, representation):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node, representation, self.tag.UPDATES)
        print("Node 1 sent average weight to {} nodes".format(len(self.clients)))

    def single_round(self, self_repr):
        """
        Runs the whole training procedure
        """
        print("Node 1 waiting for all clients to finish")
        reprs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.DONE)
        reprs.append(self_repr)
        print("Node 1 received {} clients' weights".format(len(reprs)))
        avg_wts = self.aggregate(reprs)
        self.send_representations(avg_wts)
        return avg_wts
        # wait for all clients to finish

    def assign_own_status(self, status):
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
        print(
            "Node {} is a {}, pair with {}".format(
                self.node_id, self.status, self.pair_id
            )
        )

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            # self.log_utils.logging.info("Client waiting for semaphore from {}".format(self.server_node))
            # print("Client waiting for semaphore from {}".format(self.server_node))
            status = self.comm_utils.wait_for_signal(src=0, tag=self.tag.START)
            self.assign_own_status(status)
            # print("semaphore received, start local training")
            # self.log_utils.logging.info("Client received semaphore from {}".format(self.server_node))
            if self.status == "teacher":
                self.local_train()
                # self.local_test()
                self_repr = self.get_representation()
                self.comm_utils.send_signal(
                    dest=self.pair_id, data=self_repr, tag=self.tag.DONE
                )
                print(
                    "Node {} sent repr to student node {}".format(
                        self.node_id, self.pair_id
                    )
                )
                # self.log_utils.logging.info("Client {} sending done signal to {}".format(self.node_id, self.server_node))
                # print("sending signal to node {}".format(self.server_node))
            elif self.status == "student":
                teacher_repr = self.comm_utils.wait_for_signal(
                    src=self.pair_id, tag=self.tag.DONE
                )
                print(
                    "Node {} received repr from teacher node {}".format(
                        self.node_id, self.pair_id
                    )
                )
                self.deep_mutual_train(teacher_repr)
            else:
                # self.comm_utils.send_signal(dest=self.server_node, data=self_repr, tag=self.tag.DONE)
                print("Node {} do nothing".format(self.node_id))
                # repr = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.UPDATES)

            # self.set_representation(repr)
            # test updated model
            acc = self.local_test()
            print("Node {} test_acc:{:.4f}".format(self.node_id, acc))
            self.comm_utils.send_signal(dest=0, data=acc, tag=self.tag.FINISH)


class DefKTServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def send_representations(self, representations):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node, representations, self.tag.UPDATES)
            self.log_utils.log_console(
                "Server sent {} representations to node {}".format(
                    len(representations), client_node
                )
            )
        # self.model.load_state_dict(representation)

    def test(self) -> float:
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        # TODO save the model if the accuracy is better than the best accuracy
        # so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def assigns_clients(self):
        num_teachers = self.config["num_teachers"]
        clients = list(range(1, self.num_clients + 1))
        if 2 * num_teachers > self.num_clients:
            return None  # Not enough room to pick two non-overlapping subarrays

        # Pick the starting index of the first subarray
        selected_indices = random.sample(range(self.num_clients), 2 * num_teachers)
        selected_elements = [clients[i] for i in selected_indices]

        # Divide the selected elements into two arrays of length num_teachers
        teachers = selected_elements[:num_teachers]
        students = selected_elements[num_teachers:]

        return teachers, students

    def single_round(self):
        """
        Runs the whole training procedure
        """
        teachers, students = self.assigns_clients()  # type: ignore
        self.log_utils.log_console("Teachers:{}".format(teachers))
        self.log_utils.log_console("Students:{}".format(students))
        for client_node in self.clients:
            self.log_utils.log_console(
                "Server sending status from {} to {}".format(self.node_id, client_node)
            )
            self.comm_utils.send_signal(
                dest=client_node, data=[teachers, students], tag=self.tag.START
            )

    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()

            accs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.FINISH)
            self.log_utils.log_console("Round {} done; acc {}".format(round, accs))
