import random
from collections import OrderedDict
from typing import Any, Dict, List
from torch import Tensor
from utils.communication.comm_utils import CommunicationManager
from algos.base_class import BaseClient, BaseServer
import time

# import the possible attacks
from algos.attack_add_noise import AddNoiseAttack
from algos.attack_bad_weights import BadWeightsAttack
from algos.attack_sign_flip import SignFlipAttack

from utils.gias import gia_main

import pickle

class FedAvgClient(BaseClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        self.random_params = self.model.state_dict()

    def local_test(self, **kwargs: Any):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """
        start_time = time.time()
        test_loss, test_acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device,
        )
        end_time = time.time()
        time_taken = end_time - start_time
        return [test_loss, test_acc, time_taken]


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

        # assert hasattr(self, 'images') and hasattr(self, 'labels'), "Images and labels not found"
        if "gia" in self.config and hasattr(self, 'images') and hasattr(self, 'labels'):
            # also stream image and labels
            message["images"] = self.images.to("cpu")
            message["labels"] = self.labels.to("cpu")

            message["random_params"] = self.random_params
            for key in message["random_params"].keys():
                message["random_params"][key] = message["random_params"][key].to("cpu")
    
        return message  # type: ignore

    def run_protocol(self):
        stats: Dict[str, Any] = {}
        print(f"Client {self.node_id} ready to start training")

        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]

        for round in range(start_rounds, total_rounds):
            stats["train_loss"], stats["train_acc"], stats["train_time"] = self.local_train(round)
            stats["test_loss"], stats["test_acc"], stats["test_time"] = self.local_test()
            self.local_round_done()

            self.receive_and_aggregate()
            
            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()
            
            self.log_metrics(stats=stats, iteration=round)


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
        if "gia" in self.config:
            # to store param differences for GIA attack
            self.params_s = [None for i in range(4)]
            self.params_t = [None for i in range(4)]

            # save randomly initialized parameters
            self.random_params = self.model.state_dict()

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

    def test(self, **kwargs: Any) -> List[float]:
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
        return [test_loss, test_acc, time_taken]

    def receive_and_aggregate_gia(self, round: int, attack_start_round: int, attack_end_round: int, dump_file_name: str = ""):
        reprs = self.comm_utils.all_gather()

        with open(dump_file_name, "wb") as f:
            pickle.dump(reprs, f)

        # Handle GIA-specific logic
        if "gia" in self.config:
            print("Server Running GIA attack")
            base_params = [key for key, _ in self.model.named_parameters()]
            print(base_params)

            for rep in reprs:
                client_id = rep["sender"]
                assert "images" in rep and "labels" in rep, "Images and labels not found in representation"
                model_state_dict = rep["model"]

                # Extract relevant model parameters
                model_params = OrderedDict(
                    (key, value) for key, value in model_state_dict.items()
                    if key in base_params
                )

                random_params = rep["random_params"]
                random_params = OrderedDict(
                    (key, value) for key, value in random_params.items()
                    if key in base_params
                )

                # Store parameters based on attack start and end rounds
                if round == attack_start_round:
                    self.params_s[client_id - 1] = model_params
                elif round == attack_end_round:
                    self.params_t[client_id - 1] = model_params
                    images = rep["images"]
                    labels = rep["labels"]

                    # with open(f"params_t_{client_id}.pkl", "wb") as f:
                    #     pickle.dump(model_params, f)
                    # with open(f"params_s_{client_id}.pkl", "wb") as f:
                    #     pickle.dump(self.params_s[client_id - 1], f)
                    # with open(f"random_params_{client_id}.pkl", "wb") as f:
                    #     pickle.dump(random_params, f)

                    # Launch GIA attack
                    p_s, p_t = self.params_s[client_id - 1], self.params_t[client_id - 1]
                    gia_main(p_s, p_t, base_params, self.model, labels, images, client_id)

        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)


    def receive_and_aggregate(self):
        reprs = self.comm_utils.all_gather()
        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)

    def single_round(self, round: int, attack_start_round: int = 0, attack_end_round: int = 1):
        """
        Runs the whole training procedure.
        
        Parameters:
            round (int): Current round of training.
            attack_start_round (int): The starting round to initiate the attack.
            attack_end_round (int): The last round for the attack to be performed.
        """
        # Normal training when outside the attack range

        if round < attack_start_round or round > attack_end_round:
            self.receive_and_aggregate()
        else:
            # Set file name based on start or end of attack range
            dump_file_name = ""
            if round == attack_start_round:
                dump_file_name = "/u/yshi23/sonar/src/start_reprs"
            elif round == attack_end_round:
                dump_file_name = "/u/yshi23/sonar/src/end_reprs"

            print(f"In round {round}, preparing for GIA with file: {dump_file_name}")
            self.receive_and_aggregate_gia(round, attack_start_round, attack_end_round, dump_file_name)
         

    def run_protocol(self):
        stats: Dict[str, Any] = {}
        print(f"Client {self.node_id} ready to start training")
        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]
        for round in range(start_rounds, total_rounds):

            self.local_round_done()
            self.single_round(round)
            stats["bytes_received"], stats["bytes_sent"] = self.comm_utils.get_comm_cost()
            stats["test_loss"], stats["test_acc"], stats["test_time"] = self.test()
            self.log_metrics(stats=stats, iteration=round)
