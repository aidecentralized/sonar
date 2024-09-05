iid_clients_collab_new = {
    "algo": "dare",
    "exp_id": 6,
    "exp_type": "iid_clients_collab_entropy",
    "load_existing": False,
    "start_epoch": 500,
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # Learning setup
    "num_clients": 2,
    "top_k": 1,
    "samples_per_user": 1000,
    "device_ids": {"node_0": [], "node_1": [0], "node_2": [1]},
    # top_k peers to communicate with, currently it is same as num_clients - 1 because
    # we are not including the client itself
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 64,
    # params for model
    # "method": "orig", "ismaml": 0,
    # "position": 4, "inp_shape": [0, 256, 8, 8], "out_shape": [0, 256, 8, 8],
    "method": "fast_meta",
    "ismaml": 1,
    "lr_g": 5e-3,
    "lr_z": 0.015,
    "position": 0,
    "inp_shape": [0, 256],
    "out_shape": [0, 3, 32, 32],
    # Params for gradient descent on data
    "data_lr": 0.05,
    "steps": 2000,
    "alpha_preds": 10,
    "alpha_tv": 2.5e-7,
    "alpha_l2": 0.0,
    "alpha_f": 10.0,
    "distill_batch_size": 128,
    "distill_epochs": 10,
    "warmup": 20,
    "first_time_steps": 2000,
    "exp_keys": ["distill_epochs", "steps", "position", "warmup"],
}

iid_clients_isolated_new = {
    "algo": "isolated",
    "exp_id": 6,
    "exp_type": "iid_clients_isolated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 1,
    # no concept of client in isolated learning
    "device_ids": {"node_0": [1, 2]},
    # Learning setup
    "num_clients": 1,
    "samples_per_user": 2000,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

iid_clients_federated_new = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "iid_clients_federated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [0]},
    # Learning setup
    "num_clients": 3,
    "samples_per_user": 2000,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

iid_random_clients_new = {
    "algo": "fedran",
    "exp_id": 150,
    "exp_type": "iid_random_clients_federated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/cifar10",
    # "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [0], "node_1": [0], "node_2": [1], "node_3": [1]},
    # Learning setup
    "num_clients": 3,
    "samples_per_user": 2000,
    "target_clients": 1,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

iid_weight_clients_new = {
    "algo": "fedweight",
    "exp_id": 150,
    "exp_type": "iid_weight_clients_federated",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/cifar10",
    # "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {
        "node_0": [0],
        "node_1": [0],
        "node_2": [1],
        "node_3": [1],
        "node_4": [2],
        "node_5": [2],
    },
    # Learning setup
    "num_clients": 5,
    "samples_per_user": 500,
    "target_clients": 2,
    "similarity": "CosineSimilarity",  # CosineSimilarity or EuclideanDistance
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

iid_swarm_clients_new = {
    "algo": "swarm",
    "exp_id": 150,
    "exp_type": "iid_swarm",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/cifar10",
    # "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {
        "node_0": [0],
        "node_1": [0],
        "node_2": [1],
        "node_3": [1],
        "node_4": [2],
        "node_5": [2],
    },
    # Learning setup
    "num_clients": 5,
    "samples_per_user": 500,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

iid_l2c_clients_new = {
    "algo": "l2c",
    "exp_id": 150,
    "exp_type": "iid_l2c",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/cifar10",
    # "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {
        "node_0": [4],
        "node_1": [4],
        "node_2": [5],
        "node_3": [7],
        "node_4": [7],
    },
    # Learning setup
    "num_clients": 4,
    "samples_per_user": 100,
    "alpha_lr": 0.01,
    "K_0": 1,
    "T_0": 2,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

iid_dispfl_clients_new = {
    "algo": "dispfl",
    "exp_id": 200,
    "exp_type": "iid_dispfl",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/cifar10",
    # "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {
        "node_0": [2],
        "node_1": [3],
        "node_2": [4],
        "node_3": [5],
        "node_4": [2],
    },
    # Learning setup
    "num_clients": 4,
    "samples_per_user": 500,
    "cs": "random",
    "neighbors": 2,
    "active_rate": 0.8,  # prob of active node
    "dense_ratio": 0.5,
    "erk_power_scale": 1,
    "anneal_factor": 0.5,
    "dis_gradient_check": None,
    "static": None,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 128,
    "exp_keys": [],
}

iid_defkt_clients_new = {
    "algo": "defkt",
    "exp_id": 200,
    "exp_type": "iid_defkt",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/cifar10",
    # "dpath": "./imgs/cifar10",
    "seed": 2,
    # server can have overlapping device ids with clients because
    # both are not used at the same time
    "device_ids": {"node_0": [4], "node_1": [4], "node_2": [4], "node_3": [4]},
    "num_teachers": 1,
    # Learning setup
    "num_clients": 3,
    "samples_per_user": 500,
    "dense_ratio": 0.5,
    "erk_power_scale": 1,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}


current_config = iid_clients_federated_new
