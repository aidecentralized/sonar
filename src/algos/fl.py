"""Module for FedAvg implementation"""
from collections import OrderedDict
import sys
from typing import Any, Dict, List
from torch import Tensor
from utils.communication.comm_utils import CommunicationManager
from utils.log_utils import LogUtils
from algos.base_class import BaseClient, BaseServer
import os
import time

class FedAvgClient(BaseClient):
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        try:
            config['log_path'] = f"{config['log_path']}/node_{self.node_id}"
            os.makedirs(config['log_path'])
        except FileExistsError:
            color_code = "\033[91m" # Red color
            reset_code = "\033[0m"   # Reset to default color
            print(f"{color_code}Log directory for the node {self.node_id} already exists in {config['log_path']}")
            print(f"Exiting to prevent accidental overwrite{reset_code}")
            sys.exit(1)

        config['load_existing'] = False
        self.client_log_utils = LogUtils(config)

    def local_train(self, round: int, **kwargs: Any):
        """
        Train the model locally
        """
        start_time = time.time()
        avg_loss, avg_accuracy = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device
        )
        end_time = time.time()
        time_taken = end_time - start_time

        self.client_log_utils.log_console(
            "Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(self.node_id, avg_loss, avg_accuracy, time_taken)
            )
        self.client_log_utils.log_summary("Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(self.node_id, avg_loss, avg_accuracy, time_taken))

        self.client_log_utils.log_tb(f"train_loss/client{self.node_id}", avg_loss, round)
        self.client_log_utils.log_tb(f"train_accuracy/client{self.node_id}", avg_accuracy, round)

    def local_test(self, **kwargs: Any):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        pass

    def get_representation(self, **kwargs: Any) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.state_dict() # type: ignore


    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """Set the model weights"""
        self.model.load_state_dict(representation)

    def run_protocol(self):
        """Run the FedAvg protocol for the client"""
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.local_train(round)
            self.local_test()
            repr = self.get_representation()
            self.client_log_utils.log_summary("Client {} sending done signal to {}".format(self.node_id, self.server_node))
            self.comm_utils.send(self.server_node, repr)
            self.client_log_utils.log_summary("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))
            repr = self.comm_utils.receive(self.server_node)
            self.client_log_utils.log_summary("Client {} received new model from {}".format(self.node_id, self.server_node))
            self.set_representation(repr)
            # self.client_log_utils.log_summary("Round {} done for Client {}".format(round, self.node_id))


class FedAvgServer(BaseServer):
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )
        self.folder_deletion_signal = config["folder_deletion_signal_path"]

    # def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]):
    #     # All models are sampled currently at every round
    #     # Each model is assumed to have equal amount of data and hence
    #     # coeff is same for everyone
    #     num_users = len(model_wts)
    #     coeff = 1 / num_users # this assumes each node has equal amount of data
    #     avgd_wts: OrderedDict[str, Tensor] = OrderedDict()
    #     first_model = model_wts[0]

    #     for node_num in range(num_users):
    #         local_wts = model_wts[node_num]
    #         for key in first_model.keys():
    #             if node_num == 0:
    #                 avgd_wts[key] = coeff * local_wts[key].to('cpu')
    #             else:
    #                 avgd_wts[key] += coeff * local_wts[key].to('cpu')
    #     # put the model back to the device
    #     for key in avgd_wts.keys():
    #         avgd_wts[key] = avgd_wts[key].to(self.device)
    #     return avgd_wts

    def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]):
        num_users = len(model_wts)
        coeff = 1 / num_users
        avgd_wts: OrderedDict[str, Tensor] = OrderedDict()

        for key in model_wts[0].keys():
            avgd_wts[key] = sum(coeff * m[key] for m in model_wts) # type: ignore

        # Move to GPU only after averaging
        for key in avgd_wts.keys():
            avgd_wts[key] = avgd_wts[key].to(self.device)
        return avgd_wts

    def aggregate(self, representation_list: List[OrderedDict[str, Tensor]], **kwargs: Any) -> OrderedDict[str, Tensor]:
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model
        """
        self.comm_utils.broadcast(representation)
        print("braodcasted")
        self.model.load_state_dict(representation)

    def test(self, **kwargs: Any) -> List[float]:
        """
        Test the model on the server
        """
        start_time = time.time()
        test_loss, test_acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        end_time = time.time()
        time_taken = end_time - start_time
        # TODO save the model if the accuracy is better than the best accuracy
        # so far
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return [test_loss, test_acc, time_taken]

    def single_round(self):
        """
        Runs the whole training procedure
        """
        # calculate how much memory torch is occupying right now
        # self.log_utils.log_console("Server waiting for all clients to finish")
        reprs = self.comm_utils.all_gather()
        # self.log_utils.log_console("Server received all clients done signal")

        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)
        #Remove the signal file after confirming that all client paths have been created
        if os.path.exists(self.folder_deletion_signal):
            os.remove(self.folder_deletion_signal)

    def run_protocol(self):
        self.log_utils.log_console("Starting clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):
            self.log_utils.log_console("Starting round {}".format(round))
            self.log_utils.log_summary("Starting round {}".format(round))
            self.single_round()
            self.log_utils.log_console("Server testing the model")
            loss, acc, time_taken = self.test()
            self.log_utils.log_tb(f"test_acc/clients", acc, round)
            self.log_utils.log_tb(f"test_loss/clients", loss, round)
            self.log_utils.log_console("Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(round, acc, loss, time_taken))
            # self.log_utils.log_summary("Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(round, acc, loss, time_taken))
            self.log_utils.log_console("Round {} complete".format(round))
            self.log_utils.log_summary("Round {} complete".format(round,))
