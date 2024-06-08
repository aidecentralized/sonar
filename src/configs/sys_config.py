
# System Configuration

system_config = {
    "num_users": 4,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/",
    "device_ids": {
        "iid_dispfl": {"node_0": [2], "node_1": [3], "node_2": [4], "node_3": [5], "node_4": [2]},
        "iid_defkt": {"node_0": [4], "node_1": [4], "node_2": [4], "node_3": [4]},
        "non_iid": {"node_0": [0], "node_1": [1], "node_2": [2], "node_3": [3]}
    },
    "dataset_splits": {
        "iid": {"num_clients": 4, "samples_per_client": 500},
        "non_iid": {"num_clients": 8, "samples_per_client": 32}
    }
}


def get_device_ids(algo):
    return system_config["device_ids"].get(algo, {})
