from collections import OrderedDict
from torch import Tensor
from typing import TypeAlias, Dict, List, Union, Tuple, Optional
import random

ConfigType: TypeAlias = Dict[
    str,
    Union[
        str,
        float,
        int,
        bool,
        List[str],
        List[int],
        List[float],
        List[bool],
        Tuple[Union[int, str, float, bool, None], ...],
        Optional[List[int]],
    ],
]

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