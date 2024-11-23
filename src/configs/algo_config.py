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
    "topology": {"name": "ring"}, # type: ignore
    "rounds": 2,

    # Model parameters
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


feddatarepr: ConfigType = {
    "algo": "feddatarepr",
    "num_rep": 1,
    "load_existing": False,
    # Similarity params
    "representation": "train_data",  # "test_data", "train_data", "dreams"
    "num_repr_samples": 16,
    # "CTLR_KL" Collaborator is Teacher using Learner Representation
    # "CTCR_KL" Collaborator is Teacher using Collaborator Representation - Default row
    # "LTLR_KL" Collaborator is Learner using Learner Representation - Default column
    # "CTAR_KL" Collaborator is Teacher using ALL Representations (from every other client)
    # "train_loss_inv" : 1-loss/total
    # "train_loss_sm": 1-softmax(losses)
    "similarity_metric": "train_loss_inv",
    # Memory params
    "sim_running_average": 10,
    "sim_exclude_first": (5, 5),  # (first rounds, first rounds after T0)
    # Clients selection
    "target_users_before_T_0": 0,  # feddatarepr_users-1,
    "target_users_after_T_0": 1,
    "T_0": 10,  # round after wich only target_users_after_T_0 peers are kept
    # highest, lowest, [lower_exp]_sim_sampling, top_x, xth, uniform_rdm
    "selection_strategy": "uniform_rdm",  # "uniform_rdm",
    # "eps_greedy": 0.1,
    # "num_users_top_x" : 1, # Ideally: size community-1
    # "selection_temperature": 0.5, # For all strategy with temperature
    # Consensus params
    # "sim_averaging", "sim_of_sim", "vote_1hop", "affinity_propagation_clustering", "mean_shift_clustering", "club"
    "consensus": "mean_shift_clustering",  # "affinity_propagation_clustering",
    # "affinity_precomputed": False, # If False similarity row are treated as data points and not as similarity values
    # "club_weak_link_strategy": "own_cluster_and_pointing_to", #"own_cluster_and_pointing_to", pointing_to, own_cluster
    # "vote_consensus": (2,2), #( num_voter, num_vote_per_voter)
    # "sim_consensus_top_a": 3,
    # "community_type": "dataset",
    # "num_communities": len(domainnet_classes),
    # Learning setup
    "warmup_epochs": 5,
    "epochs_per_round": 5,
    "rounds_per_selection": 1,  # Number of rounds before selecting new collaborator(s)
    "rounds": 10,
    "model": "resnet10",
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    "model_lr": 1e-4,
    "batch_size": 16,
    # Dreams params
    # "reprs_position": 0,
    # "inp_shape": [3, 32, 32] ,
    # "inv_lr": 1e-1,
    # "inv_epochs": 500,
    # "alpha_preds": 0.1,
    # "alpha_tv": 2.5e-3,
    # "alpha_l2": 1e-7,
    # "alpha_f": 10.0,
    # "dreams_keep_best": False, # Use reprs with lowest loss
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
    feddatarepr,
]

# Malicious List of algorithm configurations
malicious_algo_config_list: List[ConfigType] = [
    traditional_fl,
    malicious_traditional_data_poisoning_attack,
    malicious_traditional_model_poisoning_attack,
    malicious_traditional_model_update_attack,
]


default_config_list: List[ConfigType] = [traditional_fl]
