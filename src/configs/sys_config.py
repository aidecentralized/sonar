# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
system_config = {
    "num_users": 3,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 31,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [0], "node_1": [0],"node_2": [0], "node_3": [0]},
    "samples_per_user": 1000, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

object_detect_system_config = {
    "num_users": 5,
    "experiment_path": "./experiments/",
    "dset": "pascal",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/pascal/VOCdevkit/VOC2012/",
    "seed": 31,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [1], "node_1": [2], "node_2": [3], "node_3": [4], "node_4": [5], "node_5": [7]},
    "samples_per_user": 1000, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

current_config = object_detect_system_config