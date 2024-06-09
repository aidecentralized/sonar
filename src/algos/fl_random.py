from collections import OrderedDict
from typing import Any, Dict, List
import torch
import torch.nn as nn
import random
import numpy as np

from algos.base_class import BaseFedAvgClient, BaseFedAvgServer

from collections import defaultdict 
from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays

class FedRanClient(BaseFedAvgClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def get_collaborator_weights(self, reprs_dict, round):
        """
        Returns the weights of the collaborators for the current round
        """ 
        total_rounds = self.config["rounds"]
        within_community_sampling = self.config.get("within_community_sampling",1)
        p_within_decay = self.config.get("p_within_decay",None)
        if p_within_decay is not None:
            if p_within_decay == "linear_inc":         
                within_community_sampling = within_community_sampling * (round / total_rounds)
            elif p_within_decay == "linear_dec":
                within_community_sampling = within_community_sampling * (1 - round / total_rounds)
            elif p_within_decay == "exp_inc":
                # Alpha scaled so that it goes from p to (1-p) in R rounds
                alpha = np.log((1-within_community_sampling)/within_community_sampling)
                within_community_sampling = within_community_sampling * np.exp(alpha * round / total_rounds)
            elif p_within_decay == "exp_dec":
                # Alpha scaled so that it goes from p to (1-p) in R rounds
                alpha = np.log(within_community_sampling/(1-within_community_sampling))
                within_community_sampling = within_community_sampling * np.exp(- alpha * round / total_rounds)
            elif p_within_decay == "log_inc":
                alpha = np.exp(1/within_community_sampling)-1
                within_community_sampling = within_community_sampling * np.log2(1 + alpha * round / total_rounds)
        
        if  random.random() <= within_community_sampling or len(self.communities) == 1:
            # Consider only neighbors (clients in the same community)
            indices = [id for id in sorted(list(reprs_dict.keys())) if id in self.communities[self.node_id]]
        else:
            # Consider clients from other communities
            indices = [id for id in sorted(list(reprs_dict.keys())) if id not in self.communities[self.node_id]]      
              
        num_clients_to_select = self.config[f"target_clients_{'before' if round < self.config['T_0'] else 'after'}_T_0"]
        selected_ids = random.sample(indices, min(num_clients_to_select + 1, len(indices)))
        # Force self node id to be selected, not removed before sampling to keep sampling identic across nodes (if same seed)
        selected_ids = [self.node_id] + [id for id in selected_ids if id != self.node_id][:num_clients_to_select]
       
        collab_weights = defaultdict(lambda: 0.0)
        for idx in selected_ids:
            own_aggr_weight = self.config.get("own_aggr_weight", 1/len(selected_ids))
            
            aggr_weight_strategy = self.config.get("aggr_weight_strategy", None)
            if aggr_weight_strategy is not None:
                init_weight = 0.1
                target_weight = 0.5
                if aggr_weight_strategy == "linear":
                    target_round = total_rounds // 2
                    own_aggr_weight = 1 - (init_weight + (target_weight - init_weight) * (min(1,round / target_round)))
                elif aggr_weight_strategy == "log":
                    alpha = 0.05
                    own_aggr_weight = 1 - (init_weight + (target_weight-init_weight) * (np.log(alpha*(round/total_rounds)+1)/np.log(alpha+1)))
                else:
                    raise ValueError(f"Aggregation weight strategy {aggr_weight_strategy} not implemented")
                
            if self.node_id == 1 and idx == 1:
                print(f"Collaborator {idx} weight: {own_aggr_weight}")
            if idx == self.node_id:
                collab_weights[idx] = own_aggr_weight
            else:
                collab_weights[idx] = (1 - own_aggr_weight) / (len(selected_ids) - 1)
            
        return collab_weights
    
    def get_representation(self):
        return self.get_model_weights()
    
    def mask_last_layer(self):
        wts = self.get_model_weights()
        keys = self.model_utils.get_last_layer_keys(wts)
        key = [k for k in keys if "weight" in k][0]
        weight = torch.zeros_like(wts[key])
        weight[self.classes_of_interest] = wts[key][self.classes_of_interest]
        self.model.load_state_dict({key: weight.to(self.device)}, strict=False)

    def freeze_model_except_last_layer(self):
        wts = self.get_model_weights()
        keys = self.model_utils.get_last_layer_keys(wts)
        
        for name, param in self.model.named_parameters():
            if name not in keys:
                param.requires_grad = False
               
    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True 
            
    def flatten_repr(self,repr):
        params = []
        
        for key in repr.keys():
            params.append(repr[key].view(-1))

        params = torch.cat(params)
        
        return params
    
    def compute_pseudo_grad_norm(self, prev_wts, new_wts):
        return np.linalg.norm(self.flatten_repr(prev_wts) -  self.flatten_repr(new_wts))

    def run_protocol(self):
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError("Start round different from 0 not implemented yet")
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        for round in range(start_round, total_rounds):
            stats = {}
            
            # Wait on server to start the round
            self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.ROUND_START)
            
            if self.config.get("finetune_last_layer", False):
                self.freeze_model_except_last_layer()

            # Train locally and send the representation to the server
            if not self.config.get("local_train_after_aggr", False):
                stats["train_loss"], stats["train_acc"] = self.local_train(epochs_per_round)
            
            repr = self.get_representation()
            self.comm_utils.send_signal(dest=self.server_node, data=repr, tag=self.tag.REPR_ADVERT)

            # Collect the representations from all other nodes from the server 
            reprs = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.REPRS_SHARE)
            
            # In the future this dict might be generated by the server to send only requested models
            reprs_dict = {k:v for k,v in enumerate(reprs, 1)}
            
            # Aggregate the representations based on the collab weights
            collab_weights_dict = self.get_collaborator_weights(reprs_dict, round)
            
            # Since clients representations are also used to transmit knowledge
            # There is no need to fetch the server for the selected clients' knowledge
            models_wts = reprs_dict
            
            layers_to_ignore = self.model_keys_to_ignore
            active_collab = set([k for k,v in collab_weights_dict.items() if v > 0])
            inter_commu_last_layer_to_aggr = self.config.get("inter_commu_layer", None)
            # If partial merging is on and some client selected client is outside the community, ignore layers after specified layer
            if inter_commu_last_layer_to_aggr is not None and len(set(self.communities[self.node_id]).intersection(active_collab)) != len(active_collab):
                layer_idx = self.model_utils.models_layers_idx[self.config["model"]][inter_commu_last_layer_to_aggr]
                layers_to_ignore = self.model_keys_to_ignore + list(list(models_wts.values())[0].keys())[layer_idx+1:]
            
            avg_wts = self.weighted_aggregate(models_wts, collab_weights_dict, keys_to_ignore=layers_to_ignore)
                        
            # Average whole model by default
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

                # Test updated model 
                stats["test_acc_after_training"] = self.local_test()

            # Include collab weights in the stats
            collab_weight = np.zeros(self.config["num_clients"])
            for k,v  in collab_weights_dict.items():
                collab_weight[k-1] = v
            stats["Collaborator weights"] = collab_weight

            self.comm_utils.send_signal(dest=self.server_node, data=stats, tag=self.tag.ROUND_STATS)

class FedRanServer(BaseFedAvgServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        
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
        
        # Send signal to all clients to start local training
        for client_node in self.clients:
            self.comm_utils.send_signal(dest=client_node, data=None, tag=self.tag.ROUND_START)
        self.log_utils.log_console("Server waiting for all clients to finish local training")
                
        # Collect models from all clients
        models = self.comm_utils.wait_for_all_clients(self.clients, self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models") 
        
        # Broadcast the models to all clients
        self.send_representations(models)
        
        # Collect round stats from all clients
        clients_round_stats = self.comm_utils.wait_for_all_clients(self.clients, self.tag.ROUND_STATS) 
        self.log_utils.log_console("Server received all clients stats") 

        # Log the round stats on tensorboard except the collab weights
        self.log_utils.log_tb_round_stats(clients_round_stats, ["Collaborator weights"], self.round)

        self.log_utils.log_console(f"Round test acc before local training {[stats['test_acc_before_training'] for stats in clients_round_stats]}")
        self.log_utils.log_console(f"Round test acc after local training {[stats['test_acc_after_training'] for stats in clients_round_stats]}")

        return clients_round_stats

    def run_protocol(self):
        self.log_utils.log_console("Starting random P2P collaboration")
        start_round = self.config.get("start_round", 0)
        total_round = self.config["rounds"]

        # List of list stats per round
        stats = []
        for round in range(start_round, total_round):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))

            round_stats = self.single_round()
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1  
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)

        
