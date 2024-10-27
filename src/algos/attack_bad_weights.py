"""
This module implements the BadWeightsAttack class, which simulates an attack by corrupting 
a portion of the model weights. The corruption is done by scaling the weights with a 
specific factor ('weight') and applying this transformation to a portion of the weights 
determined by 'corrupt_portion'.

Classes:
    - BadWeightsAttack: Applies corruption to model weights based on a given configuration.

Usage:
    config = {"weight": 0.5, "corrupt_portion": 0.2}
    attack = BadWeightsAttack(config, model.state_dict())
    corrupted_weights = attack.get_representation()
"""

import random
from collections import OrderedDict
from typing import Dict
from torch import Tensor
from utils.types import ConfigType


class BadWeightsAttack:
    """
    A class that applies corruption to a portion of the model's weights by scaling them with 
    a predefined factor ('weight'). This can be used to simulate malicious attacks on the model.

    Attributes:
        state_dict (OrderedDict[str, Tensor]): A dictionary containing the model's state (weights).
        weight (float): A factor by which corrupted weights are scaled. Default is 0.
        corrupt_portion (float): The proportion of weights to corrupt. A float between 0 and 1.
    """

    def __init__(
        self, config: ConfigType, state_dict: Dict[str, Tensor]
    ) -> None:
        """
        Initializes the BadWeightsAttack class with the provided configuration and model state.

        Args:
            config (ConfigType): A configuration dictionary containing 'weight' and 
                                 'corrupt_portion'. 'weight' specifies the factor to 
                                 scale corrupted weights, and 'corrupt_portion' defines 
                                 the proportion of weights to corrupt.
            state_dict (OrderedDict[str, Tensor]): A dictionary containing the model's state 
                                                   (weights).
        """
        self.state_dict = state_dict
        self.weight = config.get("weight", 0)
        self.corrupt_portion = float(config.get("corrupt_portion", 1)) # type: ignore
        # TODO: Add conditions like start and end epochs or rounds when corruption occurs.

    def get_representation(self) -> Dict[str, Tensor]:
        """
        Returns a modified version of the model's state dictionary where 
        a portion of the weights are scaled by the 'weight' factor based 
        on the probability defined by 'corrupt_portion'.

        Returns:
            OrderedDict[str, Tensor]: The modified state dictionary with corrupted weights.
        """
        return OrderedDict(
            {
                key: (
                    val * self.weight if random.random() < self.corrupt_portion else val
                )
                for key, val in self.state_dict.items()
            }
        )
