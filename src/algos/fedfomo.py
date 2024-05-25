from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor,cat,zeros_like,numel,randperm, from_numpy
import torch
import torch.nn as nn
import random
import copy
import numpy as np
import math

from algos.base_class import BaseClient, BaseServer


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    DONE = 0 # Used to signal the server that the client is done with local training
    START = 1 # Used to signal by the server to start the current round
    UPDATES = 2 # Used to send the updates from the server to the clients
    LAST_ROUND = 3
    SHARE_MASKS = 4
    SHARE_WEIGHTS = 5
    FINISH = 6 # Used to signal the server to finish the current round


class FedFomoClient(BaseClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        self.dense_ratio = self.config["dense_ratio"] 
        self.anneal_factor = self.config["anneal_factor"]
        self.dis_gradient_check = self.config["dis_gradient_check"]
        self.server_node = 1 #leader node
        self.num_clients = config["num_clients"]
        self.neighbors = list(range(self.num_clients))
        if self.node_id ==1:
            self.clients = list(range(2, self.num_clients+1))

    def local_train(self):
        """
        Train the model locally
        
        """
        loss,acc = self.model_utils.train_mask(self.model, self.mask,self.optim,
                                          self.dloader, self.loss_fn,
                                          self.device)
        
        print("Node{} train loss: {}, train acc: {}".format(self.node_id,loss,acc))
        # loss = self.model_utils.train_mask(self.model, self.mask,self.optim,
        #                                   self.dloader, self.loss_fn,
        #                                   self.device)
        
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
        # if acc > self.best_acc:
        #     self.best_acc = acc
        #     self.model_utils.save_model(self.model, self.model_save_path)
        return test_loss,acc
        
    def get_trainable_params(self):
        param_dict= {}
        for name, param in self.model.module.named_parameters():
            param_dict[name] = param
        return param_dict
    
    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.module.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model weights
        """
        self.model.module.load_state_dict(representation)

    def fire_mask(self,masks,round, total_round):
        weights = self.get_representation()
        drop_ratio = self.anneal_factor / 2 * (1 + np.cos((round * np.pi) / total_round))
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]), 100000 * torch.ones_like(weights[name]))
            x, idx = torch.sort(temp_weights.view(-1).to(self.device))
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return new_masks, num_remove

    def regrow_mask(self, masks,  num_remove, gradient=None):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            if not self.dis_gradient_check:
                temp = torch.where(masks[name] == 0, torch.abs(gradient[name]).to(self.device), -100000 * torch.ones_like(gradient[name]).to(self.device))
                sort_temp, idx = torch.sort(temp.view(-1).to(self.device), descending=True)
                new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
            else:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
                idx = torch.multinomial( temp.flatten().to(self.device),num_remove[name], replacement=False)
                new_masks[name].view(-1)[idx]=1
        return new_masks

    def aggregate(self, cur_clnt, client_num_in_total, client_num_per_round, nei_indexs, w_per_mdls_lstrd, weights_local, w_local_mdl):
        """
        Aggregate the model weights
        """
        w_easy = copy.deepcopy(weights_local[nei_indexs])
        w_easy = np.maximum(w_easy, 0)
        w_easy = np.sum(w_easy)
        if w_easy == 0.0:
            return w_per_mdls_lstrd[cur_clnt]

        weight_tmp = np.maximum(weights_local, 0)
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_clnt])
        for k in w_tmp.keys():
            for clnt in nei_indexs:
                if clnt == cur_clnt:
                    w_tmp[k] += (w_local_mdl[k] - w_per_mdls_lstrd[cur_clnt][k]) * weight_tmp[clnt] / w_easy
                else:
                    w_tmp[k] += (w_per_mdls_lstrd[clnt][k] - w_per_mdls_lstrd[cur_clnt][k]) * weight_tmp[clnt] / w_easy

        return w_tmp

    def send_representations(self, representation):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node,
                                        representation,
                                        self.tag.UPDATES)
        print("Node 1 sent average weight to {} nodes".format(len(self.clients)))

    def screen_gradient(self):
        model = self.model
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss().to(self.device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels) = next(iter(self.dloader))
        x, labels = x.to(self.device), labels.to(self.device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient={}
        for name, param in self.model.module.named_parameters():
            gradient[name] = param.grad.to("cpu")
        
        return gradient
    
    def hamming_distance(self,mask_a, mask_b):
        dis = 0; total = 0

        for key in mask_a:
            dis += torch.sum(mask_a[key].int().to(self.device) ^ mask_b[key].int().to(self.device))
            total += mask_a[key].numel()
        return dis, total
    
    def benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, p_choose):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            p_choose[cur_clnt] = 0
            if random.random() >= 0.5:
                client_indexes = np.argsort(p_choose)[-num_clients:]
            else:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
                while cur_clnt in client_indexes:
                    client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        # self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    
    def model_difference(self,model_a, model_b):
        a = sum([torch.sum(torch.square(model_a[name] - model_b[name])) for name in model_a])
        return a
    
    def update_weight(self, curr_idx, nei_indexs, w_per_mdls_lstrd, weight_local, w_local):
        client = self.client_list[curr_idx]
        metrics = client.val_test(w_per_mdls_lstrd[curr_idx], self.val_data_local_dict[curr_idx])
        loss_cur_clnt = metrics["test_loss"]
        for nei_clnt in nei_indexs:
            if nei_clnt == curr_idx:
                metrics = client.val_test(w_local, self.val_data_local_dict[curr_idx])
            else:
                metrics = client.val_test(w_per_mdls_lstrd[nei_clnt], self.val_data_local_dict[curr_idx])

            loss_nei_clnt = metrics["test_loss"]
            params_dif = []
            for key in w_per_mdls_lstrd[curr_idx]:
                if nei_clnt == curr_idx:
                    params_dif.append((w_local[key] - w_per_mdls_lstrd[curr_idx][key]).view(-1))
                else:
                    params_dif.append((w_per_mdls_lstrd[nei_clnt][key] - w_per_mdls_lstrd[curr_idx][key]).view(-1))

            params_dif = torch.cat(params_dif)
            if torch.norm(params_dif) == 0:
                weight_local[nei_clnt] = 0.0
            else:
                weight_local[nei_clnt] = (loss_cur_clnt - loss_nei_clnt) / (torch.norm(params_dif))

        return weight_local
    
    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        self.params = self.get_trainable_params()
        self.index = self.node_id-1
        weights_locals = np.full((self.num_clients), 1.0 / self.num_clients)
        p_choose_locals = np.ones(shape=(self.num_clients))
        reprs_lstrnd = [copy.deepcopy(self.get_representation()) for i in range(self.num_clients)]
        repr_per_global = [copy.deepcopy(self.get_representation()) for i in range(self.num_clients)]
        for round in range(start_epochs, total_epochs):
            #wait for signal to start round            
            if round != 0:
                [reprs_lstrnd,masks_lstrnd]=self.comm_utils.wait_for_signal(src=0, tag=self.tag.LAST_ROUND)
            self.local_train()
            self.repr = self.get_representation()
            #share data with client 1
            nei_indexs = self.benefit_choose(round, self.index, self.num_clients,
                                                  self.config["neighbors"], p_choose_locals[self.index])
            # If not selected in full, the current clint is made up and the aggregation operation is performed
            if self.num_clients != self.config["neighbors"]:
                #when not active this round
                nei_indexs = np.append(nei_indexs, self.index)
            nei_indexs = np.sort(nei_indexs)
            print("Node {}'s neighbors index:{}".format(self.index,[i+1 for i in nei_indexs]))
            
            weights_locals = self.update_weight(self.index, nei_indexs, reprs_lstrnd, copy.deepcopy(weights_locals[self.index]), copy.deepcopy(self.repr))
            
            p_choose_locals= p_choose_locals + weights_locals

            
            new_repr = self.aggregate(self.index, self.num_clients,
                                                            self.config["neighbors"], nei_indexs,
                                                            reprs_lstrnd, copy.deepcopy(weights_locals), copy.deepcopy(self.repr))
            self.set_representation(new_repr)

            loss,acc = self.local_test()
            #print("Node {} test_loss: {} test_acc:{}".format(self.node_id, loss,acc))
            self.comm_utils.send_signal(dest=0, data=acc, tag=self.tag.FINISH)

class FedFomoServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        self.dense_ratio = self.config["dense_ratio"]
        self.num_clients = self.config["num_clients"]
        
    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.module.state_dict()
    
    def send_representations(self, representations):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node,
                                        representations,
                                        self.tag.UPDATES)
            self.log_utils.log_console("Server sent {} representations to node {}".format(len(representations),client_node))
        #self.model.module.load_state_dict(representation)

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

    def single_round(self,round, active_ths_rnd):
        """
        Runs the whole training procedure
        """
        for client_node in self.clients:
            self.log_utils.log_console("Server sending semaphore from {} to {}".format(self.node_id,
                                                                                    client_node))
            self.comm_utils.send_signal(dest=client_node, data=active_ths_rnd, tag=self.tag.START)
            if round != 0:
                self.comm_utils.send_signal(dest=client_node, data=[self.reprs,self.masks], tag=self.tag.LAST_ROUND)
                
        self.masks = self.comm_utils.wait_for_all_clients(self.clients, self.tag.SHARE_MASKS)
        self.reprs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.SHARE_WEIGHTS)
    
    def get_trainable_params(self):
        param_dict= {}
        for name, param in self.model.module.named_parameters():
            param_dict[name] = param
        return param_dict
    
    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        
        for round in range(start_epochs, total_epochs):
            self.round = round
            active_ths_rnd = np.random.choice([0, 1], size = self.num_clients, p = [1.0 - self.config["active_rate"], self.config["active_rate"]])
            self.log_utils.log_console("Starting round {}".format(round))
            
            #print("weight:",mask_pers_shared)
            self.single_round(round,active_ths_rnd)

            accs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.FINISH)
            self.log_utils.log_console("Round {} done; acc {}".format(round,accs))
