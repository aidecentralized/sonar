# System Configuration
system_config = {
    "num_users": 4,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "/data/unagi0/anakewat/imgs/",
    "seed": 2,
    "device_ids": {}
    "dataset_splits": {
        "iid": {"samples_per_user": 500},
        "non_iid": {"samples_per_user": 32}
    }
}
