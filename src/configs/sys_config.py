# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
from utils.config_utils import get_sliding_window_support

system_config = {
    "num_users": 3,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 2,
    #"support" : get_sliding_window_support(num_clients=NUM_CLIENT, num_classes=10, num_classes_per_client=4),

    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    # "device_ids": {"node_0": [0], "node_1": [0],"node_2": [0], "node_3": [0]}
    "device_ids": {"node_0": [1], "node_1": [1],"node_2": [1], "node_3": [1]},
    "samples_per_user": 500, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "non_iid", # Either "iid", "non_iid" "support" 
    "test_label_distribution": "non_iid", # Either "iid", "non_iid" "support"
    "test_samples_per_user": 200, # Only for non_iid test distribution
    #"test_samples_per_class": 100,
}

current_config = system_config