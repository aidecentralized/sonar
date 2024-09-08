from typing import TypeAlias, List, Dict, Union, Tuple

# Defining the type for configuration
ConfigType: TypeAlias = Dict[
    str,
    Union[str, float, int, bool, List[str], List[int], Tuple[Union[int, str, float, bool, None], ...]]
]

# Algorithm Configuration
iid_dispfl_clients_new: ConfigType = {
    "algo": "dispfl",
    "exp_id": 12,
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
    "exp_keys": []
}

traditional_fl: ConfigType = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "iid_clients_federated",
    "epochs": 1000,
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

fedweight_users: int = 3
fedweight: ConfigType = {
    "algo": "fedweight",
    "exp_id": "test1",
    "num_rep": 1,
    "target_users": 3,
    "similarity": "CosineSimilarity",
    "with_sim_consensus": True,
    "rounds": 210,
    "epochs_per_round": 5,
    "warmup_epochs": 50,
    "model": "resnet10",
    "local_train_after_aggr": True,
    "model_lr": 1e-4,
    "batch_size": 16,
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    "position": 0,
    "exp_keys": [],
}

defkt: ConfigType = {
    "algo": "defkt",
    "exp_id": "defkt_test9",
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
    "position": 0,
    "inp_shape": [128, 3, 32, 32],
    "exp_keys": [],
}

fedavg_object_detect: ConfigType = {
    "algo": "fedavg",
    "exp_id": "test_modular_yolo",
    "exp_type": "test",
    "epochs": 10,
    "model": "yolo",
    "model_lr": 1e-5,
    "batch_size": 8,
    "exp_keys": [],
}

fediso: ConfigType = {
    "algo": "fediso",
    "exp_id": "test3",
    "num_rep": 1,
    "rounds": 100,
    "epochs_per_round": 5,
    "model": "resnet10",
    "model_lr": 1e-4,
    "batch_size": 16,
    "position": 0,
    "exp_keys": []
}

L2C_users: int = 3
L2C: ConfigType = {
    "algo": "l2c",
    "sharing": "weights",
    "exp_id": "test3",
    "alpha_lr": 0.1,
    "alpha_weight_decay": 0.01,
    "target_users_before_T_0": 0,
    "target_users_after_T_0": round((L2C_users - 1) * 0.1),
    "T_0": 10,
    "epochs_per_round": 5,
    "warmup_epochs": 5,
    "rounds": 210,
    "model": "resnet10",
    "average_last_layer": True,
    "model_lr": 1e-4,
    "batch_size": 32,
    "weight_decay": 5e-4,
    "adapted_to_assumption": False,
    "position": 0,
    "inp_shape": [128, 3, 32, 32],
    "exp_keys": []
}

fedcentral: ConfigType = {
    "seed": 1,
    "algo": "centralized",
    "exp_id": "test5",
    "mask_last_layer": False,
    "fine_tune_last_layer": False,
    "epochs_per_round": 5,
    "rounds": 100,
    "model": "resnet10",
    "model_lr": 1e-4,
    "batch_size": 16,
    "position": 0,
    "inp_shape": [128, 3, 32, 32],
    "exp_keys": []
}

# Assigning the current config with proper type hinting
current_config: ConfigType = traditional_fl
