from collections import OrderedDict
import torch
from torch import Tensor

from torch.utils.data import DataLoader, Subset
import copy

from algos.base_class import BaseClient, BaseServer
from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    SEND_DATA = 0 # Used to signal by the server to start
    SHARE_DATA = 1
    SEND_MODEL = 2
    SHARE_MODEL = 3
    ROUND_STATS = 4 # Used to signal the server the client is done

class CentralizedCLient(BaseClient):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
        
    def local_train(self, epochs, data_loader):
        """
        Train the model locally
        """
        avg_loss, avg_acc = 0, 0
        for epoch in range(epochs):
            tr_loss, tr_acc = self.model_utils.train(self.model, self.optim,
                                          data_loader, self.loss_fn,
                                          self.device)
            avg_loss += tr_loss
            avg_acc += tr_acc

        avg_loss /= epochs
        avg_acc /= epochs

        return avg_loss, avg_acc
    
    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        test_loss, acc = self.model_utils.test(self.model,
                                               self._test_loader,
                                               self.loss_fn,
                                               self.device)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc
    
    def get_model_weights(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights (on the cpu)
        """
        return {k: v.cpu() for k, v in  self.model.module.state_dict().items()}
    
    def set_model_weights(self, model_wts: OrderedDict[str, Tensor], keys_to_ignore=[]):
        """
        Set the model weights
        """
        model_wts = copy.copy(model_wts)
        
        if len(keys_to_ignore) > 0:
            for key in keys_to_ignore:
                model_wts.pop(key)
            
        for key in model_wts.keys():
            model_wts[key] = model_wts[key].to(self.device)
        
        self.model.module.load_state_dict(model_wts, strict= len(keys_to_ignore) == 0)
        
    def mask_last_layer(self):
        wts = self.get_model_weights()
        keys = self.model_utils.get_last_layer_keys(wts)
        key = [k for k in keys if "weight" in k][0]
        weight = torch.zeros_like(wts[key])
        weight[self.classes_of_interest] = wts[key][self.classes_of_interest]
        self.model.module.load_state_dict({key: weight.to(self.device)}, strict=False)

    def freeze_model_except_last_layer(self):
        wts = self.get_model_weights()
        keys = self.model_utils.get_last_layer_keys(wts)
        
        for name, param in self.model.module.named_parameters():
            if name not in keys:
                param.requires_grad = False
               
    def unfreeze_model(self):
        for param in self.model.module.parameters():
            param.requires_grad = True 

    def run_protocol(self):                    
        self.comm_utils.send_signal(dest=self.server_node, data=self.train_indices, tag=self.tag.SEND_DATA)
        
        global_dloader = None
        if self.node_id == self.config["central_client"]:
            train_indices = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.SHARE_DATA)
            train_dset = Subset(self.dset_obj.train_dset, train_indices)
            batch_size = self.config["batch_size"]
            global_dloader = DataLoader(train_dset, batch_size=batch_size*len(self.device_ids), shuffle=True)

            
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
                    
        for round in range(start_round, total_rounds):
            round_stats = {}
            
            # Train locally
            if self.node_id == self.config["central_client"]:

                round_stats["train_loss"], round_stats["train_acc"] = self.local_train(epochs_per_round, global_dloader)
                global_model = self.get_model_weights()
                self.comm_utils.send_signal(dest=self.server_node, data=global_model, tag=self.tag.SEND_MODEL)
                self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.SHARE_MODEL)
            else:
                round_stats["train_loss"], round_stats["train_acc"] = 0.0, 0.0
                global_model = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.SHARE_MODEL)
                self.set_model_weights(global_model)

            # Test model
            if self.config.get("mask_last_layer", False):
                self.mask_last_layer()
            if self.config.get("fine_tune_last_layer", False):
                self.freeze_model_except_last_layer()
                self.local_train(1, self.dloader)
            round_stats["test_acc"] = self.local_test()
            
            if self.node_id == self.config["central_client"]:
                self.set_model_weights(global_model)
                self.unfreeze_model()
            
            # send stats to server
            self.comm_utils.send_signal(dest=self.server_node, data=round_stats, tag=self.tag.ROUND_STATS)
                                
class CentralizedServer(BaseServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(self.config["results_path"],
                                                                   self.node_id)
    def run_protocol(self):
        self.log_utils.log_console("Starting centralised learning")
        
        clients_samples = self.comm_utils.wait_for_all_clients(self.clients, self.tag.SEND_DATA)
        samples = set()
        for client_samples in clients_samples:
            samples.update(client_samples)
        samples = list(samples)
        
        self.comm_utils.send_signal(dest=self.config["central_client"], data=samples, tag=self.tag.SHARE_DATA)
        
        self.log_utils.log_console("Starting random P2P collaboration")
        start_round = self.config.get("start_round", 0)
        total_round = self.config["rounds"]

        # List of list stats per round
        stats = []
        for round in range(start_round, total_round):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))
        
            model = self.comm_utils.wait_for_signal(src=self.config["central_client"], tag=self.tag.SEND_MODEL)
            
            for client_node in self.clients:
                self.comm_utils.send_signal(dest = client_node,
                                            data = model,
                                            tag = self.tag.SHARE_MODEL)

            self.log_utils.log_console("Server waiting for all clients to finish")
            
            round_stats = self.comm_utils.wait_for_all_clients(self.clients,tag=self.tag.ROUND_STATS)
            stats.append(round_stats)
            
            print(f"Round test acc {[stats['test_acc'] for stats in round_stats]}")
            
        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1  
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)