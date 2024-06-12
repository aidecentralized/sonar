from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor,cat
import torch.nn as nn
import random
import os
from algos.base_class import BaseClient, BaseServer
import numpy as np


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    DONE = 0 # Used to signal the server that the client is done with local training
    START = 1 # Used to signal by the server to start the current round
    UPDATES = 2 # Used to send the updates from the server to the clients
    FINISH = 3 # Used to signal the server to finish the current round


class SWARMClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        self.server_node = 1 #leader node
        if self.node_id ==1:
            self.num_clients = config["num_clients"]
            self.clients = list(range(2, self.num_clients+1))
        
    def local_train(self):
        """
        Train the model locally
        """
        avg_loss = self.model_utils.train(self.model, self.optim,
                                          self.dloader, self.loss_fn,
                                          self.device)
        # print("Client {} finished training with loss {}".format(self.node_id, avg_loss))
        # self.log_utils.logger.log_tb(f"train_loss/client{client_num}", avg_loss, epoch)
    
    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc
        

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
            self.comm_utils.send_signal(client_node,
                                        representation,
                                        self.tag.UPDATES)
        print("Node 1 sent average weight to {} nodes".format(len(self.clients)))

    def single_round(self,self_repr):
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

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        test_accs = np.empty((self.num_clients, total_epochs))  # Transpose the shape

        for round in range(start_epochs, total_epochs):
            #self.log_utils.logging.info("Client waiting for semaphore from {}".format(self.server_node))
            #print("Client waiting for semaphore from {}".format(self.server_node))
            self.comm_utils.wait_for_signal(src=0, tag=self.tag.START)
            #print("semaphore received, start local training")
            # self.log_utils.logging.info("Client received semaphore from {}".format(self.server_node))
            self.local_train()
            #self.local_test()
            self_repr = self.get_representation()
            # self.log_utils.logging.info("Client {} sending done signal to {}".format(self.node_id, self.server_node))
            #print("sending signal to node {}".format(self.server_node))
            if self.node_id == 1:
                repr = self.single_round(self_repr)
            # self.log_utils.logging.info("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))
            #get all representation from server
            else:
                self.comm_utils.send_signal(dest=self.server_node, data=self_repr, tag=self.tag.DONE)
                print("Node {} waiting signal from node 1".format(self.node_id))
                repr = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.UPDATES)
                
            self.set_representation(repr)
            acc = self.local_test()
            print("Node {} test_acc:{:.4f}".format(self.node_id, acc))
            self.comm_utils.send_signal(dest=0, data=acc, tag=self.tag.FINISH)
            test_accs[self.node_id-1, round] = acc
            np.save('./test_accs.npy', test_accs)

class SWARMServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)

    def send_representations(self, representations):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node,
                                        representations,
                                        self.tag.UPDATES)
            self.log_utils.log_console("Server sent {} representations to node {}".format(len(representations),client_node))
        #self.model.load_state_dict(representation)

    def test(self) -> float:
        """
        Test the model on the server
        """
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        # TODO save the model if the accuracy is better than the best accuracy so far
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def single_round(self):
        """
        Runs the whole training procedure
        """
        for client_node in self.clients:
            self.log_utils.log_console("Server sending semaphore from {} to {}".format(self.node_id,
                                                                                    client_node))
            self.comm_utils.send_signal(dest=client_node, data=None, tag=self.tag.START)
        



    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        train_accs = np.zeros((12, 210))

        for round in range(start_epochs, total_epochs):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
            self.single_round()

            accs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.FINISH)
            self.log_utils.log_console("Round {} done; acc {}".format(round,accs))

            for i, acc in enumerate(accs):
                train_accs[i, round] = acc

        np.save('./train_accs.npy', train_accs)