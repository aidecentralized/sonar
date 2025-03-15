"""
This module implements the SignFlipAttack class, which applies a sign flip to model 
weights based on a specified probability. This can simulate a type of adversarial attack 
by flipping the sign of a portion of the model's parameters, depending on a configured flip rate.

Classes:
    - SignFlipAttack: Flips the sign of a portion of model weights according to a given 
                      configuration.

Usage:
    config = {"flip_rate": 0.3}
    attack = SignFlipAttack(config, model.state_dict())
    flipped_weights = attack.get_representation()
"""

import random
from collections import OrderedDict
from typing import Dict
from torch import Tensor
from utils.types import ConfigType


class SignFlipAttack:
    """
    A class that flips the sign of a portion of model weights based on a configured 
    flip rate. This can simulate an adversarial attack that introduces significant 
    changes to the model's weights.

    Attributes:
        state_dict (OrderedDict[str, Tensor]): A dictionary containing the model's state (weights).
        flip_rate (float): The probability that the sign of a weight will be flipped. 
                           A float between 0 and 1, where 1 means all weights are flipped.
    """

    def __init__(
        self, config: ConfigType, state_dict: Dict[str, Tensor]
    ) -> None:
        """
        Initializes the SignFlipAttack class with the provided configuration and model state.

        Args:
            config (ConfigType): A configuration dictionary that contains 'flip_rate', 
                                 which determines the probability of flipping the sign 
                                 of a weight.
            state_dict (OrderedDict[str, Tensor]): A dictionary containing the model's state 
                                                   (weights).
        """
        self.state_dict = state_dict
        self.flip_rate = float(config.get("flip_rate", 1)) # type: ignore
        # TODO: Add conditions such as target label, source label, start/end epochs, or rounds for the attack.

    def get_representation(self) -> Dict[str, Tensor]:
        """
        Returns a modified version of the model's state dictionary where the sign of 
        weights is flipped based on the probability defined by 'flip_rate'.

        Returns:
            OrderedDict[str, Tensor]: The modified state dictionary with flipped signs 
                                      for some weights.
        """
        return OrderedDict(
            {
                key: -1 * val if random.random() < self.flip_rate else val
                for key, val in self.state_dict.items()
            }
        )
