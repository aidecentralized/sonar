# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
system_config = {
    "num_users": 3,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 2,
    # node_0 is a server currently
    "device_ids": {"node_0": [5], "node_1": [5],"node_2": [5], "node_3": [2]},
    "samples_per_user": 500, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
}

current_config = system_config