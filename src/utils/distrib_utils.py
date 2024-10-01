"""
This module provides utility classes and functions for distributed learning.
"""

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DataParallel
from torch.utils.data import Subset, DataLoader
from resnet import ResNet34, ResNet18, ResNet50
from utils.data_utils import extr_noniid
from typing import Dict, Any


def load_weights(model_dir: str, model: nn.Module, client_num: int):
    """
    Load weights for the given model and client number from the specified directory.

    Args:
        model_dir (str): Directory where the model weights are stored.
        model (nn.Module): Model to load the weights into.
        client_num (int): Client number to identify which weights to load.

    Returns:
        nn.Module: Model with loaded weights.
    """
    wts = torch.load(f"{model_dir}/saved_models/c{client_num}.pt")
    model.load_state_dict(wts)
    print(f"successfully loaded checkpoint for client {client_num}")
    return model


class ServerObj:
    """
    Server object for federated learning.
    """

    def __init__(self, config: Dict[str, Any], obj: Dict[str, Any], rank: int) -> None:
        self.num_users = config["num_users"]
        self.samples_per_user = config["samples_per_user"]
        self.device = obj["device"]
        self.device_id = obj["device_id"]
        test_dataset = obj["dset_obj"].test_dset
        batch_size = config["batch_size"]
        num_channels = obj["dset_obj"].num_channels

        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        model_dict: Dict[str, Any] = {
            "ResNet18": ResNet18(num_channels),
            "ResNet34": ResNet34(num_channels),
            "ResNet50": ResNet50(num_channels),
        }
        model = model_dict[config["model"]]
        self.model = model.to(self.device)


class ClientObj:
    """
    Client object for federated learning.
    """

    def __init__(self, config: Dict[str, Any], obj: Dict[str, Any], rank: int) -> None:
        self.num_users = config["num_users"]
        self.samples_per_user = config["samples_per_user"]
        self.device = obj["device"]
        self.device_id = obj["device_id"]
        train_dataset = obj["dset_obj"].train_dset
        test_dataset = obj["dset_obj"].test_dset
        batch_size = config["batch_size"]
        lr = config["model_lr"]

        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        indices = np.random.permutation(len(train_dataset))

        optim = torch.optim.Adam
        self.model = ResNet34()
        self.model = DataParallel(self.model.to(self.device), device_ids=self.device_id)
        self.optim = optim(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        if "non_iid" in config["exp_type"]:
            perm = torch.randperm(10)
            sp = [(0, 2), (2, 4)]
            self.c_dset = extr_noniid(
                train_dataset,
                config["samples_per_user"],
                perm[sp[rank - 1][0] : sp[rank - 1][1]],
            )
        else:
            self.c_dset = Subset(
                train_dataset,
                indices[
                    (rank - 1) * self.samples_per_user : rank * self.samples_per_user
                ],
            )
        self.c_dloader = DataLoader(self.c_dset, batch_size=batch_size)
