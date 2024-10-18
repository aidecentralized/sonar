from collections import OrderedDict
from typing import Any, Dict, List
from utils.communication.comm_utils import CommunicationManager
from torch import Tensor, cat
import torch.nn as nn
from algos.base_class import BaseClient, BaseServer
import numpy as np


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """

    DONE = 0  # Used to signal the server that the client is done with local training
    START = 1  # Used to signal by the server to start the current round
    UPDATES = 2  # Used to send the updates from the server to the clients
    FINISH = 3  # Used to signal the server to finish the current round


class SWARMClient(BaseClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )
        self.server_node = 1  # leader node
        if self.node_id == 1:
            self.num_users = config["num_users"]
            self.clients = list(range(2, self.num_users + 1))

    def local_train(self) -> float:
        """
        Train the model locally
        """
        avg_loss, acc = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device
        )
        # print("Client {} finished training with loss {}".format(self.node_id, avg_loss))
        return acc
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)

    def local_test(self, **kwargs) -> float:
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

    def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]) -> OrderedDict:
        # All models are sampled currently at every round
        # Each model is assumed to have equal amount of data and hence
        # coeff is same for everyone
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
    ) -> OrderedDict:
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def send_representations(self, representation) -> None:
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send(client_node, representation, self.tag.UPDATES)
        print("Node 1 sent average weight to {} nodes".format(len(self.clients)))

    def single_round(self, self_repr) -> OrderedDict:
        """
        Runs the whole training procedure
        """
        print("Node 1 waiting for all clients to finish")
        reprs = self.comm_utils.all_gather(self.tag.DONE)
        reprs.append(self_repr)
        print("Node 1 received {} clients' weights".format(len(reprs)))
        avg_wts = self.aggregate(reprs)
        self.send_representations(avg_wts)
        return avg_wts
        # wait for all clients to finish

    def run_protocol(self) -> None:
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]

        for round in range(start_epochs, total_epochs):
            self.comm_utils.receive(node_ids=0, tag=self.tag.START)
            train_acc = self.local_train()
            print("Node {} train_acc:{:.4f}".format(self.node_id, train_acc))
            self.comm_utils.send(dest=0, data=train_acc, tag=self.tag.FINISH)

            self_repr = self.get_representation()
            if self.node_id == 1:
                repr = self.single_round(self_repr)
            else:
                self.comm_utils.send(
                    dest=self.server_node, data=self_repr, tag=self.tag.DONE
                )
                print("Node {} waiting signal from node 1".format(self.node_id))
                repr = self.comm_utils.receive(
                    node_ids=self.server_node, tag=self.tag.UPDATES
                )

            self.set_representation(repr)
            acc = self.local_test()
            print("Node {} test_acc:{:.4f}".format(self.node_id, acc))
            self.comm_utils.send(dest=0, data=acc, tag=self.tag.FINISH)


class SWARMServer(BaseServer):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def send_representations(self, representations) -> None:
        """
        Set the model
        """
        for client_node in self.users:
            self.comm_utils.send(client_node, representations, self.tag.UPDATES)
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

    def single_round(self) -> None:
        """
        Runs the whole training procedure
        """
        for client_node in self.users:
            self.log_utils.log_console(
                "Server sending semaphore from {} to {}".format(
                    self.node_id, client_node
                )
            )
            self.comm_utils.send(dest=client_node, data=None, tag=self.tag.START)

    def run_protocol(self) -> None:
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]

        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()
            train_acc = self.comm_utils.all_gather(self.tag.FINISH)
            test_acc = self.comm_utils.all_gather(self.tag.FINISH)
            self.log_utils.log_console(
                "Round {} done; train acc {}".format(round, train_acc)
            )
            self.log_utils.log_console(
                "Round {} done; test acc {}".format(round, test_acc)
            )
