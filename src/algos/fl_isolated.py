from algos.base_class import BaseClient, BaseServer
from utils.stats_utils import from_rounds_stats_per_client_per_round_to_dict_arrays
from typing import Any, Dict
from utils.communication.comm_utils import CommunicationManager


class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """

    START = 0  # Used to signal by the server to start
    DONE = 1  # Used to signal the server the client is done


class FedIsoClient(BaseClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.tag = CommProtocol
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def local_train(self, epochs):
        """
        Train the model locally
        """
        avg_loss, avg_acc = 0, 0
        for epoch in range(epochs):
            tr_loss, tr_acc = self.model_utils.train(
                self.model, self.optim, self.dloader, self.loss_fn, self.device
            )
            avg_loss += tr_loss
            avg_acc += tr_acc

        avg_loss /= epochs
        avg_acc /= epochs

        return avg_loss, avg_acc

    def local_test(self, **kwargs):
        """
        Test the model locally, not to be used in the traditional FedAvg
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
        self.comm_utils.send(dest=self.server_node, data=stats, tag=self.tag.DONE)


class FedIsoServer(BaseServer):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
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
                "Server sending semaphore from {} to {}".format(
                    self.node_id, client_node
                )
            )
            self.comm_utils.send(dest=client_node, data=None, tag=self.tag.START)

        self.log_utils.log_console("Server waiting for all clients to finish")
        stats = self.comm_utils.all_gather(self.tag.DONE)

        stats_dict = from_rounds_stats_per_client_per_round_to_dict_arrays(stats)
        stats_dict["round_step"] = 1
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)
