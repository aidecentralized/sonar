# Malicious Configuration
from utils.types import ConfigType
from typing import List, Dict

label_flipping: ConfigType = {
    "malicious_type": "sign_flip",
    "flip_rate": 0.3,  # 30% of the labels are flipped
    "target_label": 1,
    "source_labels": [0, 2, 3],
}

bad_weights: ConfigType = {
    "malicious_type": "bad_weights",
    "weight": 0,
}

additive_noise: ConfigType = {
    "malicious_type": "additive_noise",
    "noise_rate": 0.1,  # 10% of the data is noisy
    "noise_mean": 0,
    "noise_std": 1,
}

# data_poisoning: ConfigType = {
#     "malicious_type": "data_poisoning",
#     "poison_rate": 0.1,  # 10% of the data is poisoned
#     "poison_method": "label_swap",
#     "target_class": 1,
# }

# backdoor_attack: ConfigType = {
#     "target_label": 1,
#     "injection_rate": 0.2,  # 20% data injected
# }

# List of Malicious node configurations
malicious_config_list: Dict[str, ConfigType] = {
    "sign_flip": label_flipping,
    "bad_weights": bad_weights,
    "additive_noise": additive_noise,
#     data_poisoning,
#     backdoor_attack,
}
