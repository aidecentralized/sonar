"""
This module implements the AddNoiseAttack class, which applies Gaussian noise to the 
model weights as a form of attack. The noise is added probabilistically, with the 
intensity of the noise controlled by parameters provided via a configuration dictionary.

The AddNoiseAttack class can be used to simulate backdoor attacks by introducing noise 
that does not noticeably degrade model performance.

Classes:
    - AddNoiseAttack: Adds Gaussian noise to the model weights.

Usage:
    config = {"noise_rate": 0.5, "noise_mean": 0, "noise_std": 1}
    attack = AddNoiseAttack(config, model.state_dict())
    noisy_weights = attack.get_representation()
"""

import random
from collections import OrderedDict
from typing import Dict
from torch import Tensor
from utils.types import ConfigType


class AddNoiseAttack:
    """
    A class that adds Gaussian noise to model weights. This can be used as a form of attack, 
    often referred to as a 'backdoor attack', where the noise is introduced without causing 
    significant noticeable changes in model performance.
    
    Attributes:
        state_dict (OrderedDict[str, Tensor]): A dictionary containing the model's state (weights).
        noise_rate (float): The probability that noise will be added to each weight.
        noise_mean (float): The mean of the Gaussian noise.
        noise_std (float): The standard deviation of the Gaussian noise.
    """

    def __init__(
        self, config: ConfigType, state_dict: Dict[str, Tensor]
    ) -> None:
        """
        Initializes the AddNoiseAttack class with the provided configuration and model state.
        
        Args:
            config (ConfigType): A configuration dictionary containing noise parameters like
                                 'noise_rate', 'noise_mean', and 'noise_std'. Default values 
                                 are used if keys are missing.
            state_dict (OrderedDict[str, Tensor]): A dictionary containing the model's state 
                                                   (weights).
        """
        self.state_dict = state_dict
        self.noise_rate = float(config.get("noise_rate", 1)) # type: ignore
        self.noise_mean = float(config.get("noise_mean", 0)) # type: ignore
        self.noise_std = float(config.get("noise_std", 1)) # type: ignore
        # TODO: this is also often known as backdoor attack,
        # where the noise is added without noticeable differences in model performance
        # we should find ways to study this

    def get_representation(self) -> Dict[str, Tensor]:
        """
        Returns a modified version of the model's state dictionary where Gaussian noise 
        is added to weights with a probability defined by `noise_rate`.
        
        Returns:
            OrderedDict[str, Tensor]: The modified state dictionary with added noise.
        """
        return OrderedDict(
            {
                key: (
                    val + self.noise_std * random.gauss(self.noise_mean, self.noise_std)
                    if random.random() < self.noise_rate
                    else val
                )
                for key, val in self.state_dict.items()
            }
        )
