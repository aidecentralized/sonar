"""
This module implements the base class for malicious attacks BaseAttack.

Usage:
    node_id = self.node_id # base node's id
"""

import random
from typing import Any
from utils.types import ConfigType



class BaseAttack:
    """
    A base class for attacks.
    
    Attributes:
        node_id (int): The unique identifier of the node used to set the seed.
        config (ConfigType): A configuration dictionary containing attack parameters and random seed.
    """

    def __init__(
        self, node_id: int, config: ConfigType, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initializes the AddNoiseAttack class with the provided configuration and model state.
        
        Args:
            config (ConfigType): A configuration dictionary containing noise parameters like
                                 'seed', 'noise_rate', 'noise_mean', and 'noise_std'. Default values 
                                 are used if keys are missing.
        """
        self.rng = random.Random(int(config.get("seed", 20)) * int(config.get("num_users", 9)) + node_id)  # type: ignore
