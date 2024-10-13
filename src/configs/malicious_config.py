# Malicious Configuration
from utils.types import ConfigType
from typing import Dict

# Weight Update Attacks
sign_flip: ConfigType = {
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

# Model Poisoning Attacks
gradient_attack: ConfigType = {
    "malicious_type": "gradient_attack",
    "scaling_factor": 1, # Scaling factor for the gradient
    "noise_factor": 0.1, # Scaling factor for the Gaussian noise
}

backdoor_attack: ConfigType = {
    "malicious_type": "backdoor_attack",
    "target_labels": [1], # Target labels
    "additional_loss": 10, # Additional loss for the backdoor attack
}

# Data Poisoning Attacks
data_poisoning: ConfigType = {
    "malicious_type": "corrupt_data",
    # gaussian_noise, shot_noise, speckle_noise, impulse_noise, defocus_blur, 
    # gaussian_blur, motion_blur, zoom_blur, snow, fog, brightness, contrast
    # elastic_transform, pixelate, jpeg_compression, spatter, saturate, frost
    "corruption_fn": "gaussian_noise", 
    "corrupt_severity": 1,
}

# List of Malicious node configurations
malicious_config_list: Dict[str, ConfigType] = {
    "sign_flip": sign_flip,
    "bad_weights": bad_weights,
    "additive_noise": additive_noise,
    "gradient_attack": gradient_attack,
    "backdoor_attack": backdoor_attack,
    "data_poisoning": data_poisoning,
}
