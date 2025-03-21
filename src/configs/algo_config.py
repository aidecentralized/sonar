from typing import List
from utils.types import ConfigType

# Algorithm Configuration

traditional_fl: ConfigType = {
    # Collaboration setup
    "algo": "fedavg",
    "rounds": 2,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

fedstatic: ConfigType = {
    # Collaboration setup
    "algo": "fedstatic",
    "topology": {"name": "ring"}, # type: ignore
    "rounds": 200,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 64,
}

# List of algorithm configurations
algo_config_list: List[ConfigType] = [
    traditional_fl,
    fedstatic,
]

default_config_list: List[ConfigType] = [traditional_fl]
