import random
from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor
from utils.communication.comm_utils import CommunicationManager
from algos.fl import FedAvgClient, FedAvgServer
import time

# import the possible attacks
from algos.attack_add_noise import AddNoiseAttack
from algos.attack_bad_weights import BadWeightsAttack
from algos.attack_sign_flip import SignFlipAttack

class FedAvgPushClient(FedAvgClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)

    def run_protocol(self):
        stats: Dict[str, Any] = {}
        print(f"Client {self.node_id} ready to start training")

        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]

        for round in range(start_rounds, total_rounds):
            # Fetch model from the server
            self.receive_pushed_and_aggregate()

            stats["train_loss"], stats["train_acc"], stats["train_time"] = self.local_train(round)
            stats["test_loss"], stats["test_acc"], stats["test_time"] = self.local_test()

            # Send the model to the server
            self.push(self.server_node)

            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()
            
            self.log_metrics(stats=stats, iteration=round)

            self.local_round_done()


class FedAvgPushServer(FedAvgServer):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)

    def single_round(self):
        """
        Runs the whole training procedure
        """
        self.push(self.users)
        self.receive_pushed_and_aggregate()

    def receive_pushed_and_aggregate(self):
        reprs = self.comm_utils.all_gather_pushed()
        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)

    def run_protocol(self):
        stats: Dict[str, Any] = {}
        print(f"Client {self.node_id} ready to start training")
        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]
        for round in range(start_rounds, total_rounds):
            self.single_round()
            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()
            stats["test_loss"], stats["test_acc"], stats["test_time"] = self.test()
            self.log_metrics(stats=stats, iteration=round)
            self.local_round_done()
