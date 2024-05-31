from src.configs.sys_config import system_config, get_device_ids
# Algorithm Configuration

iid_dispfl_clients_new = {
    "algo": "dispfl",
    "exp_id": 200,
    "exp_type": "iid_dispfl",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": system_config["dataset_path"] + "cifar10",
    "seed": 2,
    "device_ids": get_device_ids("iid_dispfl"),
    "num_clients": system_config["dataset_splits"]["iid"]["num_clients"],
    "samples_per_client": system_config["dataset_splits"]["iid"]["samples_per_client"],
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

iid_defkt_clients_new = {
    "algo": "defkt",
    "exp_id": 200,
    "exp_type": "iid_defkt",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": system_config["dataset_path"] + "cifar10",
    "seed": 2,
    "device_ids": get_device_ids("iid_defkt"),
    "num_teachers": 1,
    "num_clients": system_config["dataset_splits"]["iid"]["num_clients"],
    "samples_per_client": system_config["dataset_splits"]["iid"]["samples_per_client"],
    "dense_ratio": 0.5,
    "erk_power_scale": 1,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": []
}

non_iid_clients = {
    "algo": "fedran",
    "dpath": system_config["dataset_path"] + "domainnet",
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "samples_per_client": system_config["dataset_splits"]["non_iid"]["samples_per_client"],
    "num_clients": system_config["dataset_splits"]["non_iid"]["num_clients"],
    "rounds": 210,
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr": True,
    "model_lr": 1e-4,
    "batch_size": 16,
    "average_last_layer": True,
    "position": 0,
    "exp_keys": []
}

current_config = iid_dispfl_clients_new
