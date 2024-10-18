"""
This module defines the MetaL2CClient and MetaL2CServer classes for a Meta-learning 
framework with collaborative weights in federated learning. It also includes the 
ModelEncoder class for encoding model weights.
"""

from typing import Any, Dict
from utils.communication.comm_utils import CommunicationManager
import math
import torch
import numpy as np

from torch import optim
from torch import nn
import torch.nn.functional as F
import copy

from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from algos.base_class import BaseFedAvgClient, BaseFedAvgServer


class ModelEncoder(nn.Module):
    """
    A neural network-based model encoder that encodes the weights of ResNet18 
    convolutional and batch normalization layers into a vector, while ignoring 
    the final linear layer.
    
    Args:
        model_dict (dict): Dictionary of model weights to be encoded.
    """
    def __init__(self, model_dict):
        super(ModelEncoder, self).__init__()
        self.init_encoder_weights(model_dict)

    def weight_key_converter(self, key: str) -> str:
        """
        Converts model weight keys by replacing periods with "_dot_".
        
        Args:
            key (str): The original weight key.
        
        Returns:
            str: The converted weight key.
        """
        return key.replace(".", "_dot_")

    def init_encoder_weights(self, model_dict: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the encoder's weights for convolutional and batch normalization layers.
        
        Args:
            model_dict (Dict[str, torch.Tensor]): Dictionary containing the model's layers and weights.
        """
        keys = model_dict.keys()
        conv_keys = [key for key in keys if "conv" in key]
        bn_weights_keys = [
            key for key in keys if "bn" in key and ("weight" in key or "bias" in key)
        ]

        self.ordered_keys = conv_keys + bn_weights_keys
        self.encoder_weights = nn.ModuleDict()

        for key in conv_keys:
            model = nn.Sequential(
                nn.Linear(
                    in_features=math.prod(model_dict[key].shape[1:]),
                    out_features=10,
                    bias=True,
                ),
                nn.Linear(in_features=10, out_features=5, bias=True),
            )
            self.encoder_weights[self.weight_key_converter(key)] = model

        for key in bn_weights_keys:
            model = nn.Sequential(
                nn.Linear(
                    in_features=model_dict[key].shape[0], out_features=10, bias=True
                ),
                nn.Linear(in_features=10, out_features=5, bias=True),
            )
            self.encoder_weights[self.weight_key_converter(key)] = model

    def forward(self, model_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the encoder, converting model weights into a vector.
        
        Args:
            model_dict (Dict[str, torch.Tensor]): Dictionary containing model weights.
        
        Returns:
            torch.Tensor: A concatenated tensor representation of the model's weights.
        """
        encoder_outs = []
        for key in self.ordered_keys:
            wts = model_dict[key]
            if "conv" in key:
                encoder_outs.append(
                    self.encoder_weights[self.weight_key_converter(key)](
                        wts.view(wts.shape[0], -1)
                    ).flatten()
                )
            else:
                encoder_outs.append(
                    self.encoder_weights[self.weight_key_converter(key)](wts)
                )

        return torch.cat(encoder_outs, dim=0)


class MetaL2CClient(BaseFedAvgClient):
    """
    A federated learning client that uses model weight encoding and knowledge 
    sharing with collaborative learning based on Meta-L2C.
    
    Args:
        config (Dict[str, Any]): Configuration parameters for the client.
        comm_utils (CommunicationManager): A communication manager for sending 
        and receiving data.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)

        self.encoder = ModelEncoder(self.get_model_weights())
        self.encoder_optim = optim.SGD(
            self.encoder.parameters(), lr=self.config["alpha_lr"]
        )

        self.model_keys_to_ignore = []
        if not self.config.get("average_last_layer", True):  # By default include last layer
            keys = self.model_utils.get_last_layer_keys(self.get_model_weights())
            self.model_keys_to_ignore.extend(keys)

        self.sharing_mode = self.config["sharing"]
        self.neighbors_ids = list(range(1, self.config["num_users"] + 1))

    def get_representation(self) -> Dict[str, torch.Tensor]:
        """
        Retrieves the representation of the client's current model weights.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model's weight 
            representations.
        """
        return self.model_utils.substract_model_weights(
            self.get_model_weights(), self.model_init
        )

    def get_knowledge_sharing_artifact(self) -> Dict[str, torch.Tensor]:
        """
        Retrieves the knowledge sharing artifact based on the sharing mode.
        
        Returns:
            Dict[str, torch.Tensor]: Knowledge sharing artifact (either model 
            updates or model weights).
        """
        if self.sharing_mode == "updates":
            return self.model_utils.substract_model_weights(
                self.prev_model, self.get_model_weights()
            )
        if self.sharing_mode == "weights":
            return self.get_model_weights()
        else:
            raise ValueError("Unknown sharing mode")

    def get_collaborator_weights(self, reprs_dict: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Computes the collaboration weights based on the representations and 
        the encoder.
        
        Args:
            reprs_dict (Dict[int, torch.Tensor]): A dictionary containing representations 
            from neighboring clients.
        
        Returns:
            Dict[int, torch.Tensor]: Collaboration weights for each client.
        """
        ids = self.neighbors_ids + [self.node_id]
        collab_weights = torch.zeros(len(ids))
        self.own_embedding = self.encoder(self.get_representation())
        for idx, idy in enumerate(ids):
            collab_weights[idx] = self.encoder(reprs_dict[idy]).dot(self.own_embedding)

        collab_weights = F.softmax(collab_weights, dim=0)

        collab_weights_dict = {idy: collab_weights[idx] for idx, idy in enumerate(ids)}

        return collab_weights_dict

    def learn_collab_weights(self, models_update_wts: Dict[int, Dict[str, torch.Tensor]],
                             collab_weights_tensor_dict: Dict[int, torch.Tensor]) -> tuple:
        """
        Learns the collaboration weights and updates them during validation.
        
        Args:
            models_update_wts (Dict[int, Dict[str, torch.Tensor]]): Weights updates from models.
            collab_weights_tensor_dict (Dict[int, torch.Tensor]): Collaboration weights.
        
        Returns:
            tuple: Validation loss and accuracy after learning the collaborative weights.
        """
        self.model.eval()
        val_loss, correct = 0, 0

        for data, target in self.val_dloader:
            data, target = data.to(self.device), target.to(self.device)

            self.encoder_optim.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)

            loss.backward()

            grad_dict = {k: v.grad for k, v in self.model.named_parameters()}

            for cw_id in collab_weights_tensor_dict.keys():
                cw_grad = 0
                for key in grad_dict.keys():
                    if key not in self.model_keys_to_ignore:
                        if self.sharing_mode == "updates":
                            cw_grad -= (models_update_wts[cw_id][key] * grad_dict[key].cpu()).sum()
                        elif self.sharing_mode == "weights":
                            cw_grad += (models_update_wts[cw_id][key] * grad_dict[key].cpu()).sum()
                        else:
                            raise ValueError("Unknown sharing mode")

                collab_weights_tensor_dict[cw_id].grad = cw_grad
                collab_weights_tensor_dict[cw_id].backward(retain_graph=True)

            self.encoder_optim.step()

            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.train()
        val_acc = correct / len(self.val_dloader.dataset)
        print(f"Node {self.node_id}'s learning collab weight, loss:{val_loss}, acc: {val_acc}")

        return val_loss, val_acc

    def filter_out_worse_neighbors(self, num_neighbors_to_keep: int, collab_weights_dict: Dict[int, torch.Tensor]) -> None:
        """
        Filters out neighbors with lower collaboration weights, keeping only 
        the top k neighbors.
        
        Args:
            num_neighbors_to_keep (int): Number of neighbors to keep.
            collab_weights_dict (Dict[int, torch.Tensor]): Dictionary of 
            collaboration weights.
        """
        if num_neighbors_to_keep <= 0:
            self.neighbors_ids = []
        else:
            sorted_neighbors = sorted(collab_weights_dict.items(), key=lambda x: x[1], reverse=True)
            self.neighbors_ids = [id for id, _ in sorted_neighbors[:num_neighbors_to_keep]]

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for this client, which includes local 
        training, knowledge sharing, and weight aggregation.
        """
        self.model_init = self.get_model_weights()
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        collab_weights_dict = {id: 1 for id in self.neighbors_ids}

        for cur_round in range(start_round, total_rounds):

            if cur_round == self.config["T_0"]:
                self.filter_out_worse_neighbors(
                    self.config["target_users_after_T_0"], collab_weights_dict
                )

            if self.sharing_mode == "updates":
                self.prev_model = self.get_model_weights()

            round_stats = {}

            avg_alpha = self.comm_utils.receive(self.server_node, tag=self.tag.ROUND_START)

            if avg_alpha is not None:
                self.encoder.load_state_dict(avg_alpha)

            round_stats["train_loss"], round_stats["train_acc"] = self.local_train(epochs_per_round)
            representation = self.get_representation()
            ks_artifact = self.get_knowledge_sharing_artifact()
            self.comm_utils.send(dest=self.server_node, data=(repr, ks_artifact), tag=self.tag.REPR_ADVERT)

            reprs = self.comm_utils.receive(self.server_node, tag=self.tag.REPRS_SHARE)
            reprs_dict = {k: rep for k, (rep, _) in enumerate(reprs, 1)}

            collab_weights_tensor_dict = self.get_collaborator_weights(reprs_dict)
            collab_weights_dict = {k: cw.item() for k, cw in collab_weights_tensor_dict.items()}

            models_update_wts = {k: ks_art for k, (_, ks_art) in enumerate(reprs, 1)}

            new_wts = self.aggregate(
                models_update_wts, collab_weights_dict, self.model_keys_to_ignore
            )

            if self.sharing_mode == "updates":
                new_wts = self.model_utils.substract_model_weights(self.prev_model, new_wts)

            self.set_model_weights(new_wts, self.model_keys_to_ignore)

            round_stats["test_acc"] = self.local_test()
            round_stats["validation_loss"], round_stats["validation_acc"] = self.learn_collab_weights(models_update_wts, collab_weights_tensor_dict)

            cws = np.zeros(self.config["num_users"])
            for idx, cw in collab_weights_dict.items():
                cws[idx - 1] = cw
            round_stats["collab_weights"] = cws

            alpha = self.encoder.state_dict()
            self.comm_utils.send(dest=self.server_node, data=(round_stats, alpha), tag=self.tag.ROUND_STATS)


class MetaL2CServer(BaseFedAvgServer):
    """
    A federated learning server that coordinates training across clients, averages 
    model updates, and aggregates collaboration weights.

    Args:
        config (Dict[str, Any]): Configuration parameters for the server.
        comm_utils (CommunicationManager): Communication manager for coordinating 
        the protocol with clients.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        
    def average_state_dicts(self, state_dicts: list) -> Dict[str, torch.Tensor]:
        """
        Averages a list of model state dictionaries.

        Args:
            state_dicts (list): List of state dictionaries to average.

        Returns:
            Dict[str, torch.Tensor]: Averaged state dictionary.
        """
        avg_model = copy.copy(state_dicts[0])

        for key in avg_model.keys():
            for model in state_dicts[1:]:
                avg_model[key] += model[key]
            avg_model[key] /= len(state_dicts)

        return avg_model

    def single_round(self, avg_alpha: Dict[str, torch.Tensor]) -> tuple:
        """
        Runs a single round of federated learning by coordinating client training, 
        gathering model representations, and aggregating collaboration weights.

        Args:
            avg_alpha (Dict[str, torch.Tensor]): The averaged model weights to send 
            to clients for the next round.

        Returns:
            tuple: The round statistics and the new averaged model weights.
        """
        for client_node in self.users:
            self.comm_utils.send(dest=client_node, data=avg_alpha, tag=self.tag.ROUND_START)
        self.log_utils.log_console("Server waiting for all clients to finish local training")

        reprs = self.comm_utils.all_gather(self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models")

        self.send_representations(reprs)

        round_stats_and_alphas = self.comm_utils.all_gather(self.tag.ROUND_STATS)
        alphas = [alpha for _, alpha in round_stats_and_alphas]
        round_stats = [stats for stats, _ in round_stats_and_alphas]

        avg_alpha = self.average_state_dicts(alphas)

        self.log_utils.log_console("Server received all clients stats")
        self.log_utils.log_tb_round_stats(round_stats, ["collab_weights"], self.round)
        self.log_utils.log_console(f"Round test acc {[stats['test_acc'] for stats in round_stats]}")

        return round_stats, avg_alpha

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the server, coordinating training 
        across multiple rounds and clients.
        """
        self.log_utils.log_console("Starting Meta L2C")
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]

        stats = []
        avg_alpha = None
        for cur_round in range(start_round, total_rounds):
            self.round = cur_round
            self.log_utils.log_console(f"Starting round {cur_round}")
                                       
            round_stats, avg_alpha = self.single_round(avg_alpha)
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)

