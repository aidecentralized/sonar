# System Configuration
system_config = {
    "num_users": 4,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/",
    "seed": 2,
    "device_ids": {"node_0": [5], "node_1": [5],"node_2": [5], "node_3": [2], "node_4": [2], "node_5": [3], "node_6": [3], "node_7": [3], "node_8": [3]},
    "dataset_splits": {"samples_per_user": 500}, #iid, same structure for non_iid
}
