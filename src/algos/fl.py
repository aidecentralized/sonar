"""Module for FedAvg implementation"""
from collections import OrderedDict
from typing import List
from torch import Tensor

from algos.base_class import BaseClient, BaseServer


class CommProtocol:
    """Communication protocol tags for the server and clients"""
    DONE = 0  # Used to signal that the client is done with the current round
    START = 1  # Used to signal by the server to start the current round
    UPDATES = 2  # Used to send the updates from the server to the clients

    @classmethod
    def get_tag(cls, tag_name):
        """Get tag value by name"""
        return getattr(cls, tag_name)


class FedAvgClient(BaseClient):
    """FedAvg Client implementation"""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.tag = CommProtocol

    def local_train(self, *args, **kwargs):
        """Train the model locally"""
        avg_loss = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device
        )
        print(f"Client {self.node_id} finished training with loss {avg_loss}")

    def local_test(self, *args, **kwargs):
        """Test the model locally, not to be used in the traditional FedAvg"""

    def get_representation(self, *args, **kwargs) -> OrderedDict[str, Tensor]:
        """Share the model weights"""
        return self.model.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """Set the model weights"""
        self.model.load_state_dict(representation)

    def run_protocol(self):
        """Run the FedAvg protocol for the client"""
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for _ in range(start_epochs, total_epochs):
            self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.START)
            self.local_train()
            self.local_test()
            representation = self.get_representation()
            self.comm_utils.send_signal(
                dest=self.server_node, data=representation, tag=self.tag.DONE
            )
            representation = self.comm_utils.wait_for_signal(
                src=self.server_node, tag=self.tag.UPDATES
            )
            self.set_representation(representation)


class FedAvgServer(BaseServer):
    """FedAvg Server implementation"""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.set_model_parameters(config)
        self.tag = CommProtocol
        self.model_save_path = f"{self.config['results_path']}/saved_models/node_{self.node_id}.pt"
        self.best_acc = 0

    def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]):
        """Perform federated averaging"""
        num_users = len(model_wts)
        coeff = 1 / num_users
        avgd_wts = OrderedDict()
        first_model = model_wts[0]

        for client_num in range(num_users):
            local_wts = model_wts[client_num]
            for key in first_model.keys():
                if client_num == 0:
                    avgd_wts[key] = coeff * local_wts[key].to(self.device)
                else:
                    avgd_wts[key] += coeff * local_wts[key].to(self.device)
        return avgd_wts
    def aggregate(self, representation_list: List[OrderedDict[str, Tensor]], *args, **kwargs):
        """Aggregate the model weights"""
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def set_representation(self, representation):
        """Set the model"""
        for client_node in self.users:
            self.comm_utils.send_signal(client_node, representation, self.tag.UPDATES)
        self.model.load_state_dict(representation)

    def test(self, *args, **kwargs) -> float:
        """Test the model on the server"""
        _, acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        if acc > self.best_acc:
            self.best_acc = acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return acc

    def single_round(self):
        """Runs the whole training procedure"""
        for client_node in self.users:
            self.log_utils.log_console(
                f"Server sending semaphore from {self.node_id} to {client_node}"
            )
            self.comm_utils.send_signal(dest=client_node, data=None, tag=self.tag.START)
        self.log_utils.log_console("Server waiting for all clients to finish")
        reprs = self.comm_utils.wait_for_all_clients(self.users, self.tag.DONE)
        self.log_utils.log_console("Server received all clients done signal")
        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)

    def run_protocol(self):
        """Run the FedAvg protocol for the server"""
        self.log_utils.log_console("Starting iid clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for epoch in range(start_epochs, total_epochs):
            self.log_utils.log_console(f"Starting round {epoch}")
            self.single_round()
            self.log_utils.log_console("Server testing the model")
            acc = self.test()
            self.log_utils.log_tb("test_acc/clients", acc, epoch)
            self.log_utils.log_console(f"round: {epoch} test_acc:{acc:.4f}")
            self.log_utils.log_console(f"Round {epoch} done")