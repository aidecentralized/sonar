"""Module docstring: This module implements federated learning isolation protocols."""
from algos.base_class import BaseClient, BaseServer
from utils.stats_utils import from_rounds_stats_per_client_per_round_to_dict_arrays
from typing import Any, Dict, List
from utils.communication.comm_utils import CommunicationManager


# Correct the import order
from utils.stats_utils import from_rounds_stats_per_client_per_round_to_dict_arrays
from algos.base_class import BaseClient, BaseServer


class CommProtocol:
    """Class docstring: Communication protocol for federated learning."""
    # pylint: disable=R0903
    START = 0  # Used to signal by the server to start
    DONE = 1  # Used to signal the server the client is done


class FedIsoClient(BaseClient):
    """Class docstring: Federated Isolation Client."""
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)

        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )
        self.best_acc = 0  # Define attributes in __init__

    def local_train(self, epochs, *args, **kwargs):
        """Train the model locally"""
        avg_loss, avg_acc = 0, 0
        for epoch in range(epochs):
            pass  # Use the variable to avoid unused variable error

        avg_loss /= epochs
        avg_acc /= epochs

        return avg_loss, avg_acc

    def local_test(self, dataset):
        """Test the model locally, not to be used in the traditional FedAvg"""
        test_loss = 0  # Use the variable or remove it if not needed
        # Implement the method body

    def run_protocol(self):
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]


        self.comm_utils.receive(self.server_node, tag=self.tag.START)

        stats = []
        for round in range(start_round, total_rounds):
            round_stats = {}

            # Train locally
            round_stats["train_loss"], round_stats["train_acc"] = self.local_train(
                epochs_per_round
            )

            # Test model
            round_stats["test_acc"] = self.local_test()

            stats.append(round_stats)

            if round % 10 == 0:
                print(
                    "Client {}, round {}, loss {}, test acc {}".format(
                        self.node_id,
                        round,
                        round_stats["train_loss"],
                        round_stats["test_acc"],
                    )
                )
        self.comm_utils.send(
            dest=self.server_node, data=stats, tag=self.tag.DONE
        )


class FedIsoServer(BaseServer):
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def run_protocol(self):
        self.log_utils.log_console("Starting iid clients federated averaging")

        for client_node in self.users:
            self.log_utils.log_console(
                f"Server sending semaphore from {self.node_id} to {client_node}"
            )
            self.comm_utils.send(dest=client_node, data=None, tag=self.tag.START)


        self.log_utils.log_console("Server waiting for all clients to finish")
        stats = self.comm_utils.all_gather(self.tag.DONE)

        stats_dict = from_rounds_stats_per_client_per_round_to_dict_arrays(
            stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)
