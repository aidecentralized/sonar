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


class DisPFLClient(BaseClient):
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

    def aggregate(self, nei_indexs, weights_lstrnd, masks_lstrnd):
        """
        Aggregate the model weights
        """
        #print("len masks:",mask_list)
        count_mask = copy.deepcopy(masks_lstrnd[self.index])
        for k in count_mask.keys():
            count_mask[k] = count_mask[k] - count_mask[k] #zero out by pruning
            for clnt in nei_indexs:
                count_mask[k] += masks_lstrnd[clnt][k].to(self.device) #mask
        for k in count_mask.keys():
            count_mask[k] = np.divide(1, count_mask[k].cpu(), out = np.zeros_like(count_mask[k].cpu()), where = count_mask[k].cpu() != 0)

        #update weight temp
        w_tmp = copy.deepcopy(weights_lstrnd[self.index])
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in self.neighbors:
                if k in self.params:
                    w_tmp[k] += from_numpy(count_mask[k]).to(self.device) * weights_lstrnd[clnt][k].to(self.device)
                else:
                    w_tmp[k] = weights_lstrnd[self.index][k]
        w_p_g = copy.deepcopy(w_tmp)
        for name in self.mask:
            w_tmp[name] = w_tmp[name] * self.mask[name].to(self.device)
        return w_tmp, w_p_g
    
    def send_representations(self, representation):
        """
        Set the model
        """
        for client_node in self.clients:
            self.comm_utils.send_signal(client_node,
                                        representation,
                                        self.tag.UPDATES)
        print("Node 1 sent average weight to {} nodes".format(len(self.clients)))

    
    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse = 0.5):
        spasities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.dense_ratio
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            print('initialize by ERK')
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
                        raw_probabilities[name] = (np.sum(params[name].shape) / np.prod(params[name].shape)
                                                  ) ** self.config["erk_power_scale"]
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if name in dense_layers:
                    spasities[name] = 0
                else:
                    spasities[name] = (1 - epsilon * raw_probabilities[name])
        return spasities
     
    def init_masks(self, params, sparsities):
        masks = OrderedDict()
        for name in params:
            masks[name] = zeros_like(params[name])
            dense_numel = int((1-sparsities[name])*numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] =1
        return masks
    
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
    
    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, dist_local, total_dist, cs = False, active_ths_rnd = None):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd==1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes==cur_clnt)[0]))
        return client_indexes
    
    def model_difference(self,model_a, model_b):
        a = sum([torch.sum(torch.square(model_a[name] - model_b[name])) for name in model_a])
        return a
    
    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        self.params = self.get_trainable_params()
        sparsities = self.calculate_sparsities(self.params,sparse = self.dense_ratio) #calculate sparsity to create masks
        self.mask = self.init_masks(self.params,sparsities)  #mask_per_local
        dist_locals = np.zeros(shape=(self.num_clients))
        self.index = self.node_id-1
        masks_lstrnd = [self.mask for i in range(self.num_clients)]
        weights_lstrnd = [copy.deepcopy(self.get_representation()) for i in range(self.num_clients)]
        w_per_globals = [copy.deepcopy(self.get_representation()) for i in range(self.num_clients)]
        for round in range(start_epochs, total_epochs):
            #wait for signal to start round            
            active_ths_rnd = self.comm_utils.wait_for_signal(src=0, tag=self.tag.START)
            if round != 0:
                [weights_lstrnd,masks_lstrnd]=self.comm_utils.wait_for_signal(src=0, tag=self.tag.LAST_ROUND)
            self.repr = self.get_representation()
            dist_locals[self.index],total_dis = self.hamming_distance(masks_lstrnd[self.index],self.mask)
            print("Node{}: local mask change {}/{}".format(self.node_id,dist_locals[self.index],total_dis))
            #share data with client 1
            if active_ths_rnd[self.index] == 0:
                nei_indexs = np.array([])
            else:
                nei_indexs = self._benefit_choose(round, self.index, self.num_clients,
                                                self.config["neighbors"], dist_locals, total_dis, self.config["cs"], active_ths_rnd)
            # If not selected in full, the current clint is made up and the aggregation operation is performed
            if self.num_clients != self.config["neighbors"]:
                #when not active this round
                nei_indexs = np.append(nei_indexs, self.index)
            print("Node {}'s neighbors index:{}".format(self.index,[i+1 for i in nei_indexs]))
            
            for tmp_idx in nei_indexs:
                if tmp_idx != self.index:
                    dist_locals[tmp_idx],_ = self.hamming_distance(self.mask, masks_lstrnd[tmp_idx])

            if self.config["cs"]!="full":
                print("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.config["cs"]))
            else:
                print("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.config["cs"]))
            if active_ths_rnd[self.index] != 0:
                nei_distances = [dist_locals[i] for i in nei_indexs]
                print("choose mask diff: " + str(nei_distances))
                
            #calculate new initial model
            if active_ths_rnd[self.index] == 1:
                new_repr, w_per_globals[self.index] = self.aggregate(nei_indexs, weights_lstrnd, masks_lstrnd)
            else: 
                new_repr = copy.deepcopy(weights_lstrnd[self.index])
                w_per_globals[self.index] = copy.deepcopy(weights_lstrnd[self.index])
            model_diff= self.model_difference(new_repr,self.repr)
            print("Node {} model_diff{}".format(self.node_id,model_diff))
            self.comm_utils.send_signal(dest=0, data=copy.deepcopy(self.mask), tag=self.tag.SHARE_MASKS)
            
            self.set_representation(new_repr)
            
            # # locally train
            print("Node {} local train".format(self.node_id))
            self.local_train()
            loss,acc = self.local_test()
            print("Node {} local test: {}".format(self.node_id,acc))
            repr = self.get_representation()
            #calculate new mask m_k,t+1
            gradient = None
            if not self.config["static"]:
                if not self.dis_gradient_check:
                    gradient = self.screen_gradient()
                self.mask, num_remove = self.fire_mask(self.mask, round, total_epochs)
                self.mask = self.regrow_mask(self.mask, num_remove, gradient)
            self.comm_utils.send_signal(dest=0, data=copy.deepcopy(repr), tag=self.tag.SHARE_WEIGHTS)
           
            #test updated model 
            self.set_representation(repr)
            loss,acc = self.local_test()
            #print("Node {} test_loss: {} test_acc:{}".format(self.node_id, loss,acc))
            self.comm_utils.send_signal(dest=0, data=acc, tag=self.tag.FINISH)

class DisPFLServer(BaseServer):
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
