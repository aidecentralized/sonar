"""
This module implements FedAssClient and FedAssServer classes for federated learning.
"""

import math
import numpy as np

from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from algos.base_class import BaseFedAvgClient, BaseFedAvgServer


class FedAssClient(BaseFedAvgClient):
    """
    FedAssClient extends BaseFedAvgClient to implement specific client-side
    functionalities for a federated learning framework.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_collaborator_weights(self, current_round):
        """
        Returns the weights of the collaborators for the current round
        """
        if self.config["strategy"] == "fixed":
            collab_weights = {
                id: 1 for id in self.config["assigned_collaborators"][self.node_id]
            }
        elif self.config["strategy"] == "direct_expo":
            power = current_round % math.floor(math.log2(self.config["num_users"] - 1))
            steps = math.pow(2, power)
            collab_id = int(((self.node_id + steps) % self.config["num_users"]) + 1)
            collab_weights = {self.node_id: 1, collab_id: 1}
        else:
            raise ValueError("Strategy not implemented")

        total = sum(collab_weights.values())
        collab_weights = {id: w / total for id, w in collab_weights.items()}
        return collab_weights

    def get_representation(self):
        """Returns the model weights as representation."""
        return self.get_model_weights()

    def run_protocol(self):
        """
        Runs the client-side federated learning protocol.
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        for current_round in range(start_round, total_rounds):
            stats = {}

            self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.ROUND_START
            )

            representation = self.get_representation()
            self.comm_utils.send_signal(
                dest=self.server_node, data=representation, tag=self.tag.REPR_ADVERT
            )

            representations = self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.REPRS_SHARE
            )

            representations_dict = dict(enumerate(representations, 1))

            collab_weights_dict = self.get_collaborator_weights(current_round)
            collaborators = [k for k, w in collab_weights_dict.items() if w > 0]

            if not (len(collaborators) == 1 and collaborators[0] == self.node_id):
                avg_weights = self.weighted_aggregate(
                    representations_dict,
                    collab_weights_dict,
                    keys_to_ignore=self.model_keys_to_ignore,
                )
                self.set_model_weights(avg_weights, self.model_keys_to_ignore)

            stats["test_acc_before_training"] = self.local_test()
            stats["train_loss"], stats["train_acc"] = self.local_train(epochs_per_round)
            stats["test_acc_after_training"] = self.local_test()

            collab_weight = np.zeros(self.config["num_users"])
            for k, v in collab_weights_dict.items():
                collab_weight[k - 1] = v
            stats["collab_weights"] = collab_weight

            self.comm_utils.send_signal(
                dest=self.server_node, data=stats, tag=self.tag.ROUND)
class FedAssServer(BaseFedAvgServer):
    """
    FedAssServer extends BaseFedAvgServer to implement specific server-side
    functionalities for a federated learning framework.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        self.best_acc = 0.0
        self.round = 0

    def test(self):
        """
        Test the model on the server.
        """
        _, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def single_round(self):
        """
        Runs a single round of the training procedure.
        """
        for client_node in self.users:
            self.comm_utils.send_signal(
                dest=client_node, data=None, tag=self.tag.ROUND_START
            )
        self.log_utils.log_console(
            "Server waiting for all clients to finish local training"
        )

        models = self.comm_utils.wait_for_all_clients(self.users, self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models")

        self.send_representations(models)
        round_stats = self.comm_utils.wait_for_all_clients(self.users, self.tag.ROUND_STATS)
        self.log_utils.log_console("Server received all clients stats")

        self.log_utils.log_tb_round_stats(round_stats, ["collab_weights"], self.round)
        self.log_utils.log_console(
            f"Round acc TALT {[stats['test_acc_after_training'] for stats in round_stats]}"
        )
        self.log_utils.log_console(
            f"Round acc TBLT {[stats['test_acc_before_training'] for stats in round_stats]}"
        )

        return round_stats

    def run_protocol(self):
        """
        Runs the server-side federated learning protocol.
        """
        self.log_utils.log_console("Starting random P2P collaboration")
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]

        stats = []
        for current_round in range(start_round, total_rounds):
            self.round = current_round
            self.log_utils.log_console(f"Starting round {current_round}")

            round_stats = self.single_round()
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)