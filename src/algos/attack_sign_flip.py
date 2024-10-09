from collections import OrderedDict
from torch import Tensor
from utils.types import ConfigType
import random


class SignFlipAttack:
    def __init__(
        self, config: ConfigType, state_dict: OrderedDict[str, Tensor]
    ) -> None:
        self.state_dict = state_dict
        self.flip_rate = float(config.get("flip_rate", 1))
        # TODO: add start and end epochs, or other conditions such as
        # target label and source labels

    def get_representation(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            {
                key: -1 * val if random.random() < self.flip_rate else val
                for key, val in self.state_dict.items()
            }
        )
