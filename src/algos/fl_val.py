import torch
import numpy as np

from algos.base_class import BaseFedAvgClient, BaseFedAvgServer

from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays


class FedValClient(BaseFedAvgClient):
    def __init__(self, config) -> None:
        super().__init__(config)

    def evaluate_model(self, model, dloader, loss_fn, device):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in dloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
        return test_loss

    def get_collaborator_similarity(self, reprs_dict):

        own = self.get_model_weights()
        collab_weights = {}
        for client_id, model in reprs_dict.items():
            self.set_model_weights(model)
            collab_weights[client_id] = self.evaluate_model(
                self.model, self.dloader, self.loss_fn, self.device
            )
        self.set_model_weights(own)

        return collab_weights

    def select_top_k(self, collab_weights, k):
        collab_similarity = {
            key: value
            for key, value in collab_weights.items()
            if key in self.communities[self.node_id]
        }

        strategy = self.config.get("selection_strategy")
        if strategy == "highest":
            sorted_collab = sorted(
                collab_similarity.items(), key=lambda item: item[1], reverse=True
            )
            selected_collab = [key for key, _ in sorted_collab if key != self.node_id][
                :k
            ]
            proba_dist = {key: 1 for key in selected_collab}
        elif strategy == "lowest":
            sorted_collab = sorted(
                collab_similarity.items(), key=lambda item: item[1], reverse=False
            )
            selected_collab = [key for key, _ in sorted_collab if key != self.node_id][
                :k
            ]
            proba_dist = {key: 1 for key in selected_collab}
        else:
            raise ValueError("Selection strategy {} not implemented".format(strategy))

        selected_collab.append(self.node_id)

        collab_weights = {
            key: 1 / len(selected_collab) if key in selected_collab else 0
            for key in collab_similarity.keys()
        }

        return collab_weights, proba_dist

    def log_clients_stats(self, client_dict, stat_name):
        clients_array = np.zeros(self.config["num_users"])
        for k, v in client_dict.items():
            clients_array[k - 1] = v
        self.round_stats[stat_name] = clients_array

    def get_representation(self):
        return self.get_model_weights()

    def run_protocol(self):
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        for round in range(start_round, total_rounds):
            self.round_stats = {}

            # Wait on server to start the round
            self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.ROUND_START
            )

            # Train locally and send the representation to the server
            if not self.config.get("local_train_after_aggr", False):
                self.round_stats["train_loss"], self.round_stats["train_acc"] = (
                    self.local_train(epochs_per_round)
                )

            repr = self.get_representation()
            self.comm_utils.send_signal(
                dest=self.server_node, data=repr, tag=self.tag.REPR_ADVERT
            )

            # Collect the representations from all other nodes from the server
            reprs = self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.REPRS_SHARE
            )

            # In the future this dict might be generated by the server to send
            # only requested models
            reprs_dict = {k: v for k, v in enumerate(reprs, 1)}

            # Aggregate the representations based on the collab weights
            sim_dict = self.get_collaborator_similarity(reprs_dict)
            self.log_clients_stats(sim_dict, "Validation loss on training data")
            num_collaborator = self.config[
                f"target_clients_{'before' if round < self.config['T_0'] else 'after'}_T_0"
            ]
            collab_weights_dict, proba_dist = self.select_top_k(
                sim_dict, num_collaborator
            )
            self.log_clients_stats(proba_dist, "Selection probability")

            # Since clients representations are also used to transmit knowledge
            # There is no need to fetch the server for the selected clients'
            # knowledge
            models_wts = reprs_dict

            avg_wts = self.weighted_aggregate(
                models_wts, collab_weights_dict, self.model_keys_to_ignore
            )

            # Average whole model by default
            self.set_model_weights(avg_wts, self.model_keys_to_ignore)

            # Train locally and send the representation to the server
            if self.config.get("local_train_after_aggr", False):
                self.round_stats["train_loss"], self.round_stats["train_acc"] = (
                    self.local_train(epochs_per_round)
                )

            # Test updated model
            self.round_stats["test_acc"] = self.local_test()

            # Include collab weights in the stats
            self.log_clients_stats(collab_weights_dict, "Collaborator weights")

            self.comm_utils.send_signal(
                dest=self.server_node, data=self.round_stats, tag=self.tag.ROUND_STATS
            )


class FedValServer(BaseFedAvgServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

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

    def single_round(self):
        """
        Runs the whole training procedure
        """

        # Send signal to all clients to start local training
        for client_node in self.users:
            self.comm_utils.send_signal(
                dest=client_node, data=None, tag=self.tag.ROUND_START
            )
        self.log_utils.log_console(
            "Server waiting for all clients to finish local training"
        )

        # Collect models from all clients
        models = self.comm_utils.wait_for_all_clients(self.users, self.tag.REPR_ADVERT)
        self.log_utils.log_console("Server received all clients models")

        # Broadcast the models to all clients
        self.send_representations(models)

        # Collect round stats from all clients
        round_stats = self.comm_utils.wait_for_all_clients(
            self.users, self.tag.ROUND_STATS
        )
        self.log_utils.log_console("Server received all clients stats")

        # Log the round stats on tensorboard except the collab weights
        # self.log_utils.log_tb_round_stats(round_stats, ["Collaborator weights"], self.round)

        self.log_utils.log_console(
            f"Round test acc {[stats['test_acc'] for stats in round_stats]}"
        )

        return round_stats

    def run_protocol(self):
        self.log_utils.log_console("Starting validation sim P2P collaboration")
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
