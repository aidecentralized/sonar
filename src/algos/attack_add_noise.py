from collections import OrderedDict
from torch import Tensor
import random
from utils.types import ConfigType

class AddNoiseAttack:
    '''
    Add a Gaussian noise to the model weights
    '''
    def __init__(self, config: ConfigType, state_dict: OrderedDict[str, Tensor]) -> None:
        self.state_dict = state_dict
        self.noise_rate = float(config.get("noise_rate", 1))
        self.noise_mean = float(config.get("noise_mean", 0))
        self.noise_std = float(config.get("noise_std", 1))
        # TODO: this is also often known as backdoor attack,
        # where the noise is added without noticeable differences in model performance
        # we should find ways to study this
        
    def get_representation(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            {key: val + self.noise_std * random.gauss(self.noise_mean, self.noise_std)
             if random.random() < self.noise_rate
             else val
             for key, val in self.state_dict.items()}
        )