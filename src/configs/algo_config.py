from typing import TypeAlias, Dict, List

ConfigType: TypeAlias = Dict[str, str|float|int|bool|List[str]|List[int]|tuple[int|str|float|bool|None]]

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
    # Learning setup
    "epochs": 1000,
    "model": "resnet10",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

fedweight_users = 3
fedweight: ConfigType = {
    "algo": "fedweight",
    "exp_id": "test1",
    "num_rep": 1,

    # Client selection
    "target_users": 3,
    "similarity": "CosineSimilarity",  # "EuclideanDistance", "CosineSimilarity",
    # "community_type": "dataset",
    "with_sim_consensus": True,
    # Learning setup
    "rounds": 210,
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
    # "own_aggr_weight": 0.3,
    # "aggr_weight_strategy": "linear",
    # params for model
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
    # params for model
    "position": 0,
    "inp_shape": [128, 3, 32, 32],
    "exp_keys": [],
}
fedavg_object_detect: ConfigType = {
    "algo": "fedavg",
    "exp_id": "test_modular_yolo",
    "exp_type": "test",
    # Learning setup
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

    # Learning setup
    "rounds": 100, 
    "epochs_per_round": 5,
    "model": "resnet10",
    "model_lr": 1e-4, 
    "batch_size": 16,

    # params for model
    #"image_size": 224,
    "position": 0, 
    #"inp_shape": [128, 3, 32, 32],

    "exp_keys": []
}

L2C_users = 3
L2C: ConfigType = {
    "algo": "l2c",
    "sharing": "weights",
    "exp_id": "test3",

    "alpha_lr": 0.1, 
    "alpha_weight_decay": 0.01,

    # Clients selection
    "target_users_before_T_0": 0, # Only used if adapted_to_assumption True otherwise all users are kept
    "target_users_after_T_0": round((L2C_users-1)*0.1),
    "T_0": 10,   # round after wich only target_users_after_T_0 peers are kept

    "epochs_per_round": 5,
    "warmup_epochs": 5,
    "rounds": 210, 
    "model": "resnet10",
    "average_last_layer": True,
    "model_lr": 1e-4, #0.01, #1e-4, 
    "batch_size": 32,
    # "optimizer": "adam",
    "weight_decay": 5e-4,
    "adapted_to_assumption": False,
    
    # params for model
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

    # params for model
    #"image_size": 128,
    "position": 0, 
    "inp_shape": [128, 3, 32, 32],
    
    "exp_keys": []
}

fedval: ConfigType = {
    "algo": "fedval",
    "exp_id": "",
    "num_rep": 1,

    # Clients selection
    "selection_strategy": "highest", # lowest,
    "target_users_before_T_0": 1,
    "target_users_after_T_0": 1,
    "T_0": 400,   # round after wich only target_users_after_T_0 peers are kept
    "community_type": None,#"dataset",
    # "num_communities": len(cifar10_rotations), #len(domainnet_classes),
    
    # Learning setup
    "rounds": 200, 
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr" : False,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,

    # params for model
    "position": 0, 
    "exp_keys": []
}

swarm_users = 3
swarm: ConfigType = {
    "algo": "swarm",
    "exp_id": "test2",
    "num_rep": 1,

    # Clients selection
    "target_users": 2,
    "similarity": "CosineSimilarity",  # "EuclideanDistance", "CosineSimilarity",
    # "community_type": "dataset",
    "with_sim_consensus": True,

    # Learning setup
    "epochs": 210,
    "rounds": 210,
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr": True,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4,
    "batch_size": 16,
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    # "own_aggr_weight": 0.3,
    # "aggr_weight_strategy": "linear",
    # params for model
    "position": 0,
    "exp_keys": [],
}

fedstatic: ConfigType = {
    "algo": "fedstatic",
    "exp_id": "test_fedtorus_5",
    "num_rep": 1,
    "topology": "torus",

    # Clients selection
    "num_users_to_select": 1,
    "leader_mode": False,
    "community_type": "dataset",
    # "within_community_sampling": 0.1,
    # "p_within_decay": "log_inc", #exp_inc, exp_dec, lin_inc, lin_dec
    # "num_communities": len(cifar10_rotations), #len(domainnet_classes),
    # Learning setup
    "rounds": 210,
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr": True,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4,
    "batch_size": 16,
    # Knowledge transfer params
    # "inter_commu_layer": "l2", # the layer until which the knowledge is transferred when collaborating outside community (within_community_sampling<1) [l1, l2, l3, l4, fc]
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    # "own_aggr_weight": 0.3,
    # "aggr_weight_strategy": "linear",
    # params for model
    "position": 0,
    "exp_keys": [],
}

current_config = fedstatic
