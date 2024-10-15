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

class FedAvgClient(BaseClient):
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.config = config

    def local_train(self, round: int, **kwargs: Any):
        """
        Train the model locally
        """
        start_time = time.time()
        avg_loss, avg_accuracy = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device, malicious_type=self.config.get("malicious_type", "normal"), config=self.config,
        )
        end_time = time.time()
        time_taken = end_time - start_time

        self.log_utils.log_console(
            "Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(
                self.node_id, avg_loss, avg_accuracy, time_taken
            )
        )
        self.log_utils.log_summary(
            "Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(
                self.node_id, avg_loss, avg_accuracy, time_taken
            )
        )

        self.log_utils.log_tb(
            f"train_loss/client{self.node_id}", avg_loss, round
        )
        self.log_utils.log_tb(
            f"train_accuracy/client{self.node_id}", avg_accuracy, round
        )

    def local_test(self, **kwargs: Any):
        """
        Test the model locally, not to be used in the traditional FedAvg
        """

    def get_model_weights(self, **kwargs: Any) -> Dict[str, Tensor]:
        """
        Overwrite the get_model_weights method of the BaseNode
        to add malicious attacks
        TODO: this should be moved to BaseClient
        """

        malicious_type = self.config.get("malicious_type", "normal")

        if malicious_type == "normal":
            return self.model.state_dict()  # type: ignore
        elif malicious_type == "bad_weights":
            # Corrupt the weights
            return BadWeightsAttack(
                self.config, self.model.state_dict()
            ).get_representation()
        elif malicious_type == "sign_flip":
            # Flip the sign of the weights, also TODO: consider label flipping
            return SignFlipAttack(
                self.config, self.model.state_dict()
            ).get_representation()
        elif malicious_type == "add_noise":
            # Add noise to the weights
            return AddNoiseAttack(
                self.config, self.model.state_dict()
            ).get_representation()
        else:
            return self.model.state_dict()  # type: ignore
        return self.model.state_dict()  # type: ignore

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model weights
        """
        self.model.load_state_dict(representation)

    def run_protocol(self):
        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]

        for round in range(start_rounds, total_rounds):
            self.local_train(round)
            self.local_test()
            self.local_round_done()

            repr = self.comm_utils.receive([self.server_node])[0]
            self.set_representation(repr)
            # self.client_log_utils.log_summary("Round {} done for Client {}".format(round, self.node_id))


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
        self, representation_list: List[OrderedDict[str, Tensor]], **kwargs: Any
    ) -> OrderedDict[str, Tensor]:
        """
        Aggregate the model weights
        """
        avg_wts = self.fed_avg(representation_list)
        return avg_wts

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

    def single_round(self):
        """
        Runs the whole training procedure
        """
        reprs = self.comm_utils.all_gather()
        avg_wts = self.aggregate(reprs)
        self.set_representation(avg_wts)

    def run_protocol(self):
        self.log_utils.log_console("Starting clients federated averaging")
        start_rounds = self.config.get("start_rounds", 0)
        total_rounds = self.config["rounds"]
        for round in range(start_rounds, total_rounds):
            self.log_utils.log_console("Starting round {}".format(round))
            self.log_utils.log_summary("Starting round {}".format(round))
            self.local_round_done()
            self.single_round()
            self.log_utils.log_console("Server testing the model")
            loss, acc, time_taken = self.test()
            self.log_utils.log_tb("test_acc/clients", acc, round)
            self.log_utils.log_tb("test_loss/clients", loss, round)
            self.log_utils.log_console(
                "Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(
                    round, acc, loss, time_taken
                )
            )
            # self.log_utils.log_summary("Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(round, acc, loss, time_taken))
            self.log_utils.log_console("Round {} complete".format(round))
            self.log_utils.log_summary(
                "Round {} complete".format(
                    round,
                )
            )
