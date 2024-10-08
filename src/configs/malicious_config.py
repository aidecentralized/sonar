# Malicious Configuration
from utils.types import ConfigType
from typing import List

label_flipping: ConfigType = {
    "flip_rate": 0.3,  # 30% of the labels are flipped
    "target_label": 1,
    "source_labels": [0, 2, 3],
}

bad_weights: ConfigType = {
    "weight": 0,
}

data_poisoning: ConfigType = {
    "poison_rate": 0.1,  # 10% of the data is poisoned
    "poison_method": "label_swap",
    "target_class": 1,
}

backdoor_attack: ConfigType = {
    "target_label": 1,
    "injection_rate": 0.2,  # 20% data injected
}

# List of Malicious node configurations
malicious_config_list: List[ConfigType] = [
    label_flipping,
    bad_weights,
    data_poisoning,
    backdoor_attack,
]
