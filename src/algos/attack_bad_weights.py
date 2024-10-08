from collections import OrderedDict
from torch import Tensor
from utils.types import ConfigType
import random

class BadWeightsAttack:
    def __init__(self, config: ConfigType, state_dict: OrderedDict[str, Tensor]) -> None:
        self.state_dict = state_dict
        self.weight = config.get("weight", 0)
        self.corrupt_portion = float(config.get("corrupt_portion", 1))
        # TODO: add start and end epochs, or other conditions such as what rounds

    def get_representation(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            {key: val * self.weight
             if random.random() < self.corrupt_portion
             else val
             for key, val in self.state_dict.items()}
        )