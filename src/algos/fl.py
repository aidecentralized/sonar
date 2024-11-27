import random
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from torch import Tensor
from utils.communication.comm_utils import CommunicationManager
from algos.base_class import BaseClient, BaseServer
import time

# import the possible attacks
from algos.attack_add_noise import AddNoiseAttack
from algos.attack_bad_weights import BadWeightsAttack
from algos.attack_sign_flip import SignFlipAttack

class FedAvgClient(BaseClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config

    def local_test(self, **kwargs: Any) -> Tuple[float, float, float]:
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        start_time = time.time()
        test_loss, test_acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device,
        )
        end_time = time.time()
        time_taken = end_time - start_time

        self.stats["test_loss"], self.stats["test_acc"], self.stats["test_time"] = test_loss, test_acc, time_taken

        return test_loss, test_acc, time_taken


    def get_model_weights(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Overwrite the get_model_weights method of the BaseNode
        to add malicious attacks
        TODO: this should be moved to BaseClient
        """

        message = {"sender": self.node_id, "round": self.round}

        malicious_type = self.config.get("malicious_type", "normal")

        if malicious_type == "normal":
            message["model"] = self.model.state_dict()  # type: ignore
        elif malicious_type == "bad_weights":
            # Corrupt the weights
            message["model"] = BadWeightsAttack(
                self.config, self.model.state_dict()
            ).get_representation()
        elif malicious_type == "sign_flip":
            # Flip the sign of the weights, also TODO: consider label flipping
            message["model"] = SignFlipAttack(
                self.config, self.model.state_dict()
            ).get_representation()
        elif malicious_type == "add_noise":
            # Add noise to the weights
            message["model"] = AddNoiseAttack(
                self.config, self.model.state_dict()
            ).get_representation()
        else:
            message["model"] = self.model.state_dict()  # type: ignore

        # move the model to cpu before sending
        for key in message["model"].keys():
            message["model"][key] = message["model"][key].to("cpu")
    
        return message  # type: ignore

    def run_protocol(self):
        print(f"Client {self.node_id} ready to start training")

        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]

        for round in range(start_rounds, total_rounds):
            self.round_init()

            self.local_train(round)
            self.local_test()
            self.local_round_done()
            self.receive_and_aggregate()

            self.round_finalize()


class FedAvgServer(BaseServer):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )

    def fed_avg(self, model_wts: List[OrderedDict[str, Tensor]]):
        num_users = len(model_wts)
        coeff = 1 / num_users
        avgd_wts: OrderedDict[str, Tensor] = OrderedDict()
        for key in model_wts[0].keys():
            avgd_wts[key] = sum(coeff * m[key] for m in model_wts)  # type: ignore

        # Move to GPU only after averaging
        for key in avgd_wts.keys():
            avgd_wts[key] = avgd_wts[key].to(self.device)
        return avgd_wts

    def aggregate(
        self, representation_list: List[OrderedDict[str, Any]], **kwargs: Any
    ) -> OrderedDict[str, Tensor]:
        """
        Aggregate the model weights
        """
        representation_list, _ = self.strip_empty_models(representation_list)
        if len(representation_list) > 0:
            senders = [rep["sender"] for rep in representation_list if "sender" in rep]
            rounds = [rep["round"] for rep in representation_list if "round" in rep]
            for i in range(len(representation_list)):
                representation_list[i] = representation_list[i]["model"]

            avg_wts = self.fed_avg(representation_list)
            return avg_wts
        else:
            self.log_utils.log_console("No clients participated in this round. Maintaining model.")
            return self.model.state_dict()

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model
        """
        self.model.load_state_dict(representation)

    def test(self, **kwargs: Any) -> Tuple[float, float, float]:
        """
        Test the model on the server
        """
        start_time = time.time()
        test_loss, test_acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device,
        )
        end_time = time.time()
        time_taken = end_time - start_time
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.model_utils.save_model(self.model, self.model_save_path)
        
        self.stats["test_loss"], self.stats["test_acc"], self.stats["test_time"] = test_loss, test_acc, time_taken
        return test_loss, test_acc, time_taken

    def receive_and_aggregate(self):
        reprs = self.comm_utils.all_gather()
        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)

    def single_round(self):
        """
        Runs the whole training procedure
        """
        self.receive_and_aggregate()

    def run_protocol(self):
        print(f"Client {self.node_id} ready to start training")
        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]
        for round in range(start_rounds, total_rounds):
            self.round_init()

            self.local_round_done()
            self.single_round()
            self.test()

            self.round_finalize()
