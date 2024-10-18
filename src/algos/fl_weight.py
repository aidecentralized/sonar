from collections import OrderedDict
from typing import Any, Dict
from utils.communication.comm_utils import CommunicationManager
from torch import Tensor, cat
import torch.nn as nn
import numpy as np
from algos.base_class import BaseFedAvgClient, BaseFedAvgServer
from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays


class FedWeightClient(BaseFedAvgClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config

        self.with_sim_consensus = self.config.get("with_sim_consensus", False)

    def get_representation(self) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.get_model_weights()

    def flatten_repr(self, repr):
        params = []

        for key in repr.keys():
            params.append(repr[key].view(-1))

        params = cat(params)

        return params

    def cal_similarity(self, representations: Dict[int, OrderedDict[str, Tensor]]):
        """
        Returns a list of similarity between their own node's weights and
        the other nodes' weights
        """
        similarity = self.config["similarity"]
        if similarity == "CosineSimilarity":
            sim_func = nn.CosineSimilarity(dim=0)
        elif similarity == "EuclideanDistance":

            def sim_func(x, y):
                return ((x - y) ** 2).sum(axis=0)

        else:
            raise NotImplementedError

        sim_dict = {}

        self_rep = representations[self.node_id]
        model1 = self.flatten_repr(self_rep)
        sim_dict = {
            id: float(sim_func(model1, self.flatten_repr(repr)))
            for id, repr in representations.items()
        }

        # Normalize the similarity score to [0,1]
        if similarity == "CosineSimilarity":
            sim_dict = {id: (sim + 1) / 2 for id, sim in sim_dict.items()}
        elif similarity == "EuclideanDistance":
            sim_dict = {id: 1 / max(1, dist) for id, dist in sim_dict.items()}

        return sim_dict

    def get_k_higest_sim(self, sim_dict, k):
        """
        Returns a list of k randomly selected items from lst, with the
        item at index idx included in the selection if specified.
        """

        if self.with_sim_consensus:
            own_dict = sim_dict[self.node_id]
            new_dict = {id: 0 for id in own_dict.keys()}
            tot = 0
            for c_id, c_dict in sim_dict.items():
                c_conf = own_dict[c_id]
                for c1_id, c1_score in c_dict.items():
                    # Does not take client's own similarity into account
                    if c_id == c1_id:
                        new_dict[c_id] += c_conf * own_dict[c_id]
                    else:
                        new_dict[c_id] += c_conf * c1_score
                tot += c_conf
            new_dict = {k: v / tot for k, v in new_dict.items()}
            sim_dict = new_dict

            self.log_clients_stats(sim_dict, "Consensus similarity")

        sorted_items = sorted(sim_dict.items(), key=lambda i: i[1], reverse=True)

        selected_users = [x[0] for x in sorted_items if x[0] != self.node_id][:k]
        selected_users.append(self.node_id)

        collaborator_dict = {
            client_id: (1 / len(selected_users) if client_id in selected_users else 0)
            for client_id in sim_dict.keys()
        }

        return collaborator_dict

    def log_clients_stats(self, client_dict, stat_name):
        users_array = np.zeros(self.config["num_users"])
        for k, v in client_dict.items():
            users_array[k - 1] = v
        self.round_stats[stat_name] = users_array

    def run_protocol(self):
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]

        for round in range(start_round, total_rounds):

            self.round_stats = {}

            self.comm_utils.receive(node_ids=self.server_node, tag=self.tag.ROUND_START)
            repr = self.get_representation()

            warmup = self.config["warmup_epochs"]
            if round == start_round and warmup > 0:
                loss, acc = self.local_train(warmup)
                print(f"Client {self.node_id} warmup loss {loss} acc {acc}")

            self.comm_utils.send(
                dest=self.server_node, data=repr, tag=self.tag.REPR_ADVERT
            )
            reprs = self.comm_utils.receive(
                node_ids=self.server_node, tag=self.tag.REPRS_SHARE
            )
            reprs_dict = {k: v for k, v in enumerate(reprs, 1)}

            # calculate a list of similarity between each node
            reprs_similarity = self.cal_similarity(reprs_dict)
            self.log_clients_stats(reprs_similarity, self.config["similarity"])

            if self.with_sim_consensus:
                self.comm_utils.send(
                    dest=self.server_node,
                    data=reprs_similarity,
                    tag=self.tag.C_SELECTION,
                )
                reprs_similarity = self.comm_utils.receive(
                    node_ids=self.server_node, tag=self.tag.KNLDG_SHARE
                )

            # Select K users that have the highest similarity compared to
            # their own node
            collab_weights_dict = self.get_k_higest_sim(
                reprs_similarity, k=self.config["target_users"]
            )
            self.log_clients_stats(collab_weights_dict, "Collaborator weights")

            models_wts = reprs_dict
            avg_wts = self.aggregate(
                models_wts, collab_weights_dict, self.model_keys_to_ignore
            )

            self.set_model_weights(avg_wts, self.model_keys_to_ignore)

            self.round_stats["test_acc_before_training"] = self.local_test()

            self.round_stats["train_loss"], self.round_stats["train_acc"] = (
                self.local_train(epochs_per_round)
            )

            self.round_stats["test_acc_after_training"] = self.local_test()

            self.comm_utils.send(
                dest=self.server_node, data=self.round_stats, tag=self.tag.ROUND_STATS
            )


class FedWeightServer(BaseFedAvgServer):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config

        self.with_sim_consensus = self.config.get("with_sim_consensus", False)

    def single_round(self):
        """
        Runs the whole training procedure
        """

        # Send signal to all users to start local training
        for client_node in self.users:
            self.comm_utils.send(dest=client_node, data=None, tag=self.tag.ROUND_START)
        self.log_utils.log_console(
            "Server waiting for all users to finish local training"
        )

        # Collect models from all users
        models = self.comm_utils.all_gather(self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all users models")

        # Broadcast the models to all users
        self.send_representations(models)

        if self.with_sim_consensus:
            sim_dicts = self.comm_utils.all_gather(self.tag.C_SELECTION)
            sim_dicts = {k: v for k, v in enumerate(sim_dicts, 1)}

            for client_node in self.users:
                self.comm_utils.send(
                    dest=client_node, data=sim_dicts, tag=self.tag.KNLDG_SHARE
                )

        # Collect round stats from all users
        round_stats = self.comm_utils.all_gather(self.tag.ROUND_STATS)
        self.log_utils.log_console("Server received all users stats")

        # Log the round stats on tensorboard except the collab weights
        # self.log_utils.log_tb_round_stats(round_stats, ["Collaborator weights"], self.round)

        self.log_utils.log_console(
            f"Round test_acc_before_training {[stats['test_acc_before_training'] for stats in round_stats]}"
        )
        self.log_utils.log_console(
            f"Round test_acc_after_training {[stats['test_acc_after_training'] for stats in round_stats]}"
        )

        return round_stats

    def run_protocol(self):
        self.log_utils.log_console("Starting model weight collaboration")
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
