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
        print(f"Client {self.node_id} ready to start training")

        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]

        for round in range(start_rounds, total_rounds):
            self.round_init()

            # Fetch model from the server
            self.receive_pushed_and_aggregate()
            self.local_train(round)
            self.local_test()
            # Send the model to the server
            self.push(self.server_node)
            self.local_round_done()

            self.round_finalize()


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
        print(f"Client {self.node_id} ready to start training")
        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]
        for round in range(start_rounds, total_rounds):
            self.round_init()

            self.single_round()
            self.test()
            self.local_round_done()

            self.round_finalize()
