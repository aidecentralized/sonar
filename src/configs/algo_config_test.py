#!/usr/bin/env python3

from utils.types import ConfigType

fedstatic: ConfigType = {
    # Collaboration setup
    "algo": "fedstatic",
    "topology": {"name": "watts_strogatz", "k": 3, "p": 0.2}, # type: ignore
    "rounds": 5,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

# default_config_list: List[ConfigType] = [fedstatic, fedstatic, fedstatic, fedstatic]