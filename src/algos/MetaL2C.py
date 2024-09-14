from typing import Any, Dict, List
from utils.communication.comm_utils import CommunicationManager
import math
import torch
import numpy as np
from torch import Tensor, cat, tensor, optim
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from algos.base_class import BaseFedAvgClient, BaseFedAvgServer


# Takes RestNet18 weights and returns a encoded vector
# Encode conv layers and batch norm layers, last linear layer is ignored
# by default
class ModelEncoder(nn.Module):

    def __init__(self, model_dict):
        super(ModelEncoder, self).__init__()
        self.init_encoder_weights(model_dict)

    def weight_key_converter(self, key):
        return key.replace(".", "_dot_")

    def init_encoder_weights(self, model_dict):
        keys = model_dict.keys()
        conv_keys = [key for key in keys if "conv" in key]
        bn_weights_keys = [
            key for key in keys if "bn" in key and ("weight" in key or "bias" in key)
        ]

        self.ordered_keys = conv_keys + bn_weights_keys
        self.encoder_weights = nn.ModuleDict()

        total_weights = 0
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
            total_weights += math.prod(model_dict[key].shape[1:]) * 10 + 10 * 5

        for key in bn_weights_keys:
            model = nn.Sequential(
                nn.Linear(
                    in_features=model_dict[key].shape[0], out_features=10, bias=True
                ),
                nn.Linear(in_features=10, out_features=5, bias=True),
            )
            self.encoder_weights[self.weight_key_converter(key)] = model
            total_weights += model_dict[key].shape[0] * 10 + 10 * 5

    def forward(self, model_dict):
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

    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)

        self.encoder = ModelEncoder(self.get_model_weights())
        self.encoder_optim = optim.SGD(
            self.encoder.parameters(), lr=self.config["alpha_lr"]
        )

        self.model_keys_to_ignore = []
        if not self.config.get(
            "average_last_layer", True
        ):  # By default include last layer
            keys = self.model_utils.get_last_layer_keys(self.get_model_weights())
            self.model_keys_to_ignore.extend(keys)

        self.sharing_mode = self.config["sharing"]
        self.neighbors_ids = list(range(1, self.config["num_users"] + 1))

    def get_representation(self):
        repr = self.model_utils.substract_model_weights(
            self.get_model_weights(), self.model_init
        )
        return repr

    def get_knowledge_sharing_artifact(self):
        if self.sharing_mode == "updates":
            return self.model_utils.substract_model_weights(
                self.prev_model, self.get_model_weights()
            )
        elif self.sharing_mode == "weights":
            return self.get_model_weights()
        else:
            raise ValueError("Unknown sharing mode")

    def get_collaborator_weights(self, reprs_dict):
        """
        Computes the collab weights based on the representations and the encoder
        """

        ids = self.neighbors_ids + [self.node_id]
        collab_weights = torch.zeros(len(ids))
        self.own_embedding = self.encoder(self.get_representation())
        for idx, id in enumerate(ids):
            collab_weights[idx] = self.encoder(reprs_dict[id]).dot(self.own_embedding)

        collab_weights = F.softmax(collab_weights, dim=0)

        collab_weights_dict = {}
        for idx, id in enumerate(ids):
            collab_weights_dict[id] = collab_weights[idx]

        return collab_weights_dict

    def learn_collab_weights(self, models_update_wts, collab_weights_tensor_dict):
        self.model.eval()
        val_loss, correct = 0, 0

        for data, target in self.val_dloader:
            data, target = data.to(self.device), target.to(self.device)

            self.encoder_optim.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)

            # Compute grad for weight of aggregated model
            loss.backward()

            # Create dict params(layer) name -> gradiant
            grad_dict = {k: v.grad for k, v in self.model.named_parameters()}

            for cw_id in collab_weights_tensor_dict.keys():
                cw_grad = 0
                for key in grad_dict.keys():
                    if key not in self.model_keys_to_ignore:
                        if self.sharing_mode == "updates":
                            cw_grad -= (
                                models_update_wts[cw_id][key] * grad_dict[key].cpu()
                            ).sum()
                        elif self.sharing_mode == "weights":
                            cw_grad += (
                                models_update_wts[cw_id][key] * grad_dict[key].cpu()
                            ).sum()
                        else:
                            raise ValueError("Unknown sharing mode")

                # TODO Probably better way of doing this with a single backward
                # pass
                collab_weights_tensor_dict[cw_id].grad = cw_grad
                collab_weights_tensor_dict[cw_id].backward(retain_graph=True)

            # w = self.encoder.encoder_weights[self.encoder.weight_key_converter(self.encoder.ordered_keys[0])]
            # print(f"Encoder grad: {w.grad}, {w.grad_fn}")

            self.encoder_optim.step()

            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are
            # the same
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.train()
        val_acc = correct / len(self.val_dloader.dataset)
        print(
            "Node {}'s learning collab weight, loss:{}, acc: {}".format(
                self.node_id, val_loss, val_acc
            )
        )

        return val_loss, val_acc

    def filter_out_worse_neighbors(self, num_neighbors_to_keep, collab_weights_dict):
        """
        Keep only the top k neighbors
        """
        if num_neighbors_to_keep <= 0:
            self.neighbors_ids = []
        else:
            collab_weights_dict.sort(key=lambda x: x[1], reverse=True)
            self.neighbors_ids = [
                id for id, _ in collab_weights_dict[:num_neighbors_to_keep]
            ]

    def run_protocol(self):
        self.model_init = self.get_model_weights()
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        collab_weights_dict = {id: 1 for id in self.neighbors_ids}
        for round in range(start_round, total_rounds):

            if round == self.config["T_0"]:
                self.filter_out_worse_neighbors(
                    self.config["target_users_after_T_0"], collab_weights_dict
                )

            if self.sharing_mode == "updates":
                self.prev_model = self.get_model_weights()

            round_stats = {}

            # Wait on server to start the round
            avg_alpha = self.comm_utils.receive(
                self.server_node, tag=self.tag.ROUND_START
            )
            # Load result of AllReduce of previous round
            if avg_alpha is not None:
                self.encoder.load_state_dict(avg_alpha)

            # Train locally and send the representation to the server
            # Send the knowledge sharing artifact at the same time to save
            # communication rounds
            round_stats["train_loss"], round_stats["train_acc"] = self.local_train(
                epochs_per_round
            )
            repr = self.get_representation()
            ks_artifact = self.get_knowledge_sharing_artifact()
            self.comm_utils.send(
                dest=self.server_node,
                data=(repr, ks_artifact),
                tag=self.tag.REPR_ADVERT,
            )

            # Collect the representations from all other nodes from the server
            reprs = self.comm_utils.receive(
                self.server_node, tag=self.tag.REPRS_SHARE
            )
            reprs_dict = {k: rep for k, (rep, _) in enumerate(reprs, 1)}

            # Aggregate the representations based on the collab wheigts
            collab_weights_tensor_dict = self.get_collaborator_weights(reprs_dict)
            collab_weights_dict = {
                k: cw.item() for k, cw in collab_weights_tensor_dict.items()
            }

            # Knowledge sharing artifact are received together with the
            # representations to save communication rounds
            models_update_wts = {k: ks_art for k, (_, ks_art) in enumerate(reprs, 1)}

            new_wts = self.weighted_aggregate(
                models_update_wts, collab_weights_dict, self.model_keys_to_ignore
            )

            if self.sharing_mode == "updates":
                new_wts = self.model_utils.substract_model_weights(
                    self.prev_model, new_wts
                )

            # Average whole model by default
            self.set_model_weights(new_wts, self.model_keys_to_ignore)

            # Test updated model
            round_stats["test_acc"] = self.local_test()

            # if self.node_id == 1:
            #     print(list(self.encoder.parameters())[0])
            round_stats["validation_loss"], round_stats["validation_acc"] = (
                self.learn_collab_weights(models_update_wts, collab_weights_tensor_dict)
            )
            # if self.node_id == 1:
            #     print(list(self.encoder.parameters())[0])

            cws = np.zeros(self.config["num_users"])
            for id, cw in collab_weights_dict.items():
                cws[id - 1] = cw
            round_stats["collab_weights"] = cws

            # AllReduce is computed by the server
            alpha = self.encoder.state_dict()
            self.comm_utils.send(
                dest=self.server_node,
                data=(round_stats, alpha),
                tag=self.tag.ROUND_STATS,
            )


class MetaL2CServer(BaseFedAvgServer):

    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def average_state_dicts(self, state_dicts):
        avg_model = copy.copy(state_dicts[0])

        for key in avg_model.keys():
            for model in state_dicts[1:]:
                avg_model[key] += model[key]
            avg_model[key] /= len(state_dicts)

        return avg_model

    def single_round(self, avg_alpha):
        """
        Runs the whole training procedure
        """

        # Send signal to all clients to start local training
        for client_node in self.users:
            self.comm_utils.send(
                dest=client_node, data=avg_alpha, tag=self.tag.ROUND_START
            )
        self.log_utils.log_console(
            "Server waiting for all clients to finish local training"
        )

        # Collect representations (from all clients
        reprs = self.comm_utils.all_gather(self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models")

        # Broadcast the representations to all clients
        self.send_representations(reprs)

        # Collect round stats from all clients
        round_stats_and_alphas = self.comm_utils.all_gather(self.tag.ROUND_STATS)
        alphas = [alpha for _, alpha in round_stats_and_alphas]
        round_stats = [stats for stats, _ in round_stats_and_alphas]

        # Reduce alphas
        avg_alpha = self.average_state_dicts(alphas)

        self.log_utils.log_console("Server received all clients stats")

        # Log the round stats on tensorboard
        self.log_utils.log_tb_round_stats(round_stats, ["collab_weights"], self.round)

        self.log_utils.log_console(
            f"Round test acc {[stats['test_acc'] for stats in round_stats]}"
        )

        return round_stats, avg_alpha

    def run_protocol(self):
        self.log_utils.log_console("Starting Meta L2C")
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]

        stats = []
        avg_alpha = None
        for round in range(start_round, total_rounds):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))

            round_stats, avg_alpha = self.single_round(avg_alpha)
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)
