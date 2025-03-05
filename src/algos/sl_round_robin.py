"""
Module for Split Learning Round Robin (SLRR) algorithm.
"""
from typing import Any, Dict
from utils.communication.comm_utils import CommunicationManager
import torch

from algos.base_class import BaseNode

class CommTags:
    """
    Tags for communication.
    """
    START = "start"
    END = "end"
    MODEL = "model"
    GRAD = "grad"
    ACTS = "acts"
    LABELS = "labels"
    BYE = "bye"

# SLRR -> Split Learning Round Robin
class SLRRClientNode(BaseNode):
    """
    Federated Static Client Class.
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        

    def get_representation(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Returns the activations and the labels of the client.
        """
        raise NotImplementedError("get_representation not implemented")

    def single_training_step(self, data: Any) -> None:
        # Then the client should do a forward pass on the data and 
        # send the activations and labels to the server
        # The client should then wait for the server to send the gradients
        # The client performs backward pass after getting the gradients
        raise NotImplementedError("single_training_step not implemented")

    def wait_for_signal(self) -> str | Dict[str, torch.Tensor]:
        """
        Waits for the server to send a signal that is typically
        a bye signal or model weights.
        """
        ten_minutes = 60 * 10
        message = self.comm_utils.receive_pushed(num_tries=20, time_to_wait=ten_minutes)[0]
        if message.get("model") is not None:
            return message["model"]
        elif message.get("signal") is not None:
            return message["signal"]
        else:
            raise ValueError("Invalid message received", message.keys())

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        server_signal = self.wait_for_signal()
        while server_signal != CommTags.BYE:
            self.load_weights(server_signal)
            # Wait for the server to send the starting signal
            # After the signal is received, check if it sent weights or bye signal
            # If it is a bye signal, then the client should stop
            # If it is weights, then the client should replace its local weights
            # with the received weights
            for i in range(num_epochs):
                for data in train_data_loader:
                    single_training_step(data)

            for data in test_data_loader:
                # inference API
                get_predictions(data)

            # compute loss and accuracy
            # log it and blah blah

            send_round_end_signal()   
            server_signal = wait_for_signal()



class SLRRServer(BaseNode):
    """
    Federated Static Server Class. It does not do anything.
    It just exists to keep the code compatible across different algorithms.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        pass

    def send_start_signal(self, client_num: int) -> None:
        self.comm_utils.send(client_num, self.get_model_weights())

    def send_bye(self, client_num: int) -> None:
        self.comm_utils.send(client_num, {"signal": CommTags.BYE})

    def run_protocol(self) -> None:
        for round in range(num_rounds):
            for client_num in range(1, num_clients):
                # Send the starting signal to all the clients
                self.send_start_signal(client_num)
                status, acts, labels = wait_for_signal()
                while status != CommTags.END:
                    grads = local_train(acts, labels)
                    send_gradients(grads)
                    status, acts, labels = wait_for_signal()
                else:
                    continue
        for client_num in range(1, num_clients):
            self.send_bye(client_num)
