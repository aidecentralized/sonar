from typing import Dict, List
from .malicious_config import malicious_config_list
import random
from utils.types import ConfigType


def get_malicious_types(malicious_config_list: List[ConfigType]) -> Dict[str, str]:
    """
    Assign a random malicious type to a single node.
    """
    malicious_type = random.choice(malicious_config_list)
    return malicious_type # type: ignore


# Algorithm Configuration

iid_dispfl_clients_new: ConfigType = {
    "algo": "dispfl",
    "exp_type": "iid_dispfl",
    "neighbors": 2,
    "active_rate": 0.8,
    "dense_ratio": 0.5,
    "erk_power_scale": 1,
    "anneal_factor": 0.5,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 128,
}

traditional_fl: ConfigType = {
    # Collaboration setup
    "algo": "fedavg",
    "rounds": 2,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

test_fl_inversion: ConfigType = {
    # Collaboration setup
    "algo": "fedavg",
    "rounds": 5,
    "optimizer": "sgd",
    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    # "batch_size": 256,
    "gia": True,
}

fedweight: ConfigType = {
    "algo": "fedweight",
    "num_rep": 1,
    # Client selection
    "target_users": 3,
    "similarity": "CosineSimilarity",  # "EuclideanDistance", "CosineSimilarity",
    # "community_type": "dataset",
    "with_sim_consensus": True,
    # Learning setup
    "rounds": 10,
    "epochs_per_round": 5,
    "warmup_epochs": 50,
    "model": "resnet10",
    "local_train_after_aggr": True,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4,
    "batch_size": 16,
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    # params for model
    "position": 0,
}

defkt: ConfigType = {
    "algo": "defkt",
    "central_client": 1,
    "mask_last_layer": False,
    "fine_tune_last_layer": False,
    "epochs_per_round": 5,
    "rounds": 10,
    "epochs": 10,
    "model": "resnet10",
    "model_lr": 1e-4,
    "batch_size": 16,
    "num_teachers": 1,
    # params for model
    "position": 0,
    "inp_shape": [128, 3, 32, 32],  # This should be a List[int]
}

fedavg_object_detect: ConfigType = {
    "algo": "fedavg",
    "exp_type": "",
    # Learning setup
    "epochs": 10,
    "model": "yolo",
    "model_lr": 1e-5,
    "batch_size": 8,
}

fediso: ConfigType = {
    "algo": "fediso",
    "num_rep": 1,
    # Learning setup
    "rounds": 100,
    "epochs_per_round": 5,
    "model": "resnet10",
    "model_lr": 1e-4,
    "batch_size": 16,
    # params for model
    "position": 0,
}

L2C_users: int = 3
L2C: ConfigType = {
    "algo": "l2c",
    "sharing": "weights",
    "alpha_lr": 0.1,
    "alpha_weight_decay": 0.01,
    # Clients selection
    "target_users_before_T_0": 0,  # Only used if adapted_to_assumption True otherwise all users are kept
    "target_users_after_T_0": round((L2C_users - 1) * 0.1),
    "T_0": 10,  # round after which only target_users_after_T_0 peers are kept
    "epochs_per_round": 5,
    "warmup_epochs": 5,
    "rounds": 210,
    "model": "resnet10",
    "average_last_layer": True,
    "model_lr": 1e-4,
    "batch_size": 32,
    "weight_decay": 5e-4,
    "adapted_to_assumption": False,
    # params for model
    "position": 0,
    "inp_shape": [128, 3, 32, 32],  # This should be a List[int]
}

fedcentral: ConfigType = {
    "algo": "centralized",
    "mask_last_layer": False,
    "fine_tune_last_layer": False,
    "epochs_per_round": 5,
    "rounds": 100,
    "model": "resnet10",
    "model_lr": 1e-4,
    "batch_size": 16,
    # params for model
    "position": 0,
    "inp_shape": [128, 3, 32, 32],
}

fedval: ConfigType = {
    "algo": "fedval",
    "num_rep": 1,
    # Clients selection
    "selection_strategy": "highest",  # lowest,
    "target_users_before_T_0": 1,
    "target_users_after_T_0": 1,
    "T_0": 400,  # round after which only target_users_after_T_0 peers are kept
    "community_type": None,  # "dataset",
    # Learning setup
    "rounds": 200,
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr": False,
    "model_lr": 1e-4,
    "batch_size": 16,
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    # params for model
    "position": 0,
}

swarm_users: int = 3
swarm: ConfigType = {
    "algo": "swarm",
    "num_rep": 1,
    # Clients selection
    "target_users": 2,
    "similarity": "CosineSimilarity",  # "EuclideanDistance", "CosineSimilarity",
    "with_sim_consensus": True,
    # Learning setup
    "epochs": 210,
    "rounds": 210,
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr": True,
    "model_lr": 1e-4,
    "batch_size": 16,
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    # params for model
    "position": 0,
}

fedstatic: ConfigType = {
    # Collaboration setup
    "algo": "fedstatic",
    "topology": {"name": "watts_strogatz", "k": 3, "p": 0.2}, # type: ignore
    # "topology": {"name": "base_graph", "max_degree": 2}, # type: ignore
    "rounds": 3,
    # Model parameters
    "optimizer": "sgd", # TODO comment out for real training
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

swift: ConfigType = {
    # Collaboration setup
    "algo": "swift",
    "topology": {"name": "watts_strogatz", "k": 3, "p": 0.2}, # type: ignore
    "rounds": 20,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

fed_dynamic_weights: ConfigType = {
    # Collaboration setup
    "algo": "feddynamic",
    # comparison describes the metric or algorithm used to compare the weights of the models
    # sampling describes the method used to sample the neighbors after the comparison
    "topology": {"comparison": "weights_l2", "sampling": "closest"}, # type: ignore
    "rounds": 20,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

fed_dynamic_loss: ConfigType = {
    # Collaboration setup
    "algo": "feddynamic",
    "topology": {"comparison": "loss", "sampling": "closest"}, # type: ignore
    "rounds": 20,

    # Model parameters
    "model": "resnet6",
    "model_lr": 3e-4,
    "batch_size": 256,
}

fedavgpush: ConfigType = {
    # Collaboration setup
    "algo": "fedavgpush",
    "rounds": 2,

    # Model parameters
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
}

metaL2C_cifar10: ConfigType = {
    "algo": "metal2c",
    "sharing": "weights",  # "updates"
    # Client selection
    "target_users_before_T_0": 0,
    "target_users_after_T_0": 1,
    "K_0": 0,  # number of peers to keep as neighbors at T_0 (!) inverse that in L2C paper
    "T_0": 250,  # round after wich only K_0 peers are kept
    "alpha_lr": 0.1,
    "alpha_weight_decay": 0.01,
    "epochs_per_round": 5,
    "rounds": 3,
    "model": "resnet18",
    "average_last_layer": False,
    "model_lr": 1e-4,
    "batch_size": 64,
    "optimizer": "sgd",
    "weight_decay": 5e-4,
    # params for model
    "position": 0,
    "inp_shape": [128, 3, 32, 32],
}


# Malicious Algorithm Configuration
malicious_traditional_model_update_attack: ConfigType = {
    **traditional_fl,
    **malicious_config_list["bad_weights"],
}

malicious_traditional_data_poisoning_attack: ConfigType = {
    **traditional_fl,
    **malicious_config_list["data_poisoning"],
}

malicious_traditional_model_poisoning_attack: ConfigType = {
    **traditional_fl,
    **malicious_config_list["backdoor_attack"],
}



# List of algorithm configurations
algo_config_list: List[ConfigType] = [
    iid_dispfl_clients_new,
    traditional_fl,
    malicious_traditional_data_poisoning_attack,
    malicious_traditional_model_poisoning_attack,
    malicious_traditional_model_update_attack,
    fedweight,
    defkt,
    fedavg_object_detect,
    fediso,
    L2C,
    fedcentral,
    fedval,
    swarm,
    fedstatic,
    metaL2C_cifar10,
]

# Malicious List of algorithm configurations
malicious_algo_config_list: List[ConfigType] = [
    traditional_fl,
    malicious_traditional_data_poisoning_attack,
    malicious_traditional_model_poisoning_attack,
    malicious_traditional_model_update_attack,
]

default_config_list: List[ConfigType] = [traditional_fl]
# default_config_list: List[ConfigType] = [fedstatic, fedstatic, fedstatic, fedstatic]
