# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
from utils.config_utils import get_sliding_window_support, get_device_ids

def get_domain_support(num_clients, base, domains):
    assert num_clients % len(domains) == 0

    clients_per_domain = num_clients // len(domains)
    support = {}
    support["0"] = f"{base}_{domains[0]}"
    for i in range(1, num_clients + 1):
        support[str(i)] = f"{base}_{domains[(i-1) // clients_per_domain]}"
    return support

DOMAINNET_DMN = ["real", "sketch", "clipart"]

def get_domainnet_support(num_clients, domains=DOMAINNET_DMN):
    return get_domain_support(num_clients, "domainnet", domains)

domainnet_base_dir = "/u/abhi24/matlaberp2/p2p/imgs/domainnet/"
domainnet_dpath = {
    "domainnet_real": domainnet_base_dir,
    "domainnet_sketch": domainnet_base_dir,
    "domainnet_clipart": domainnet_base_dir,
    "domainnet_infograph": domainnet_base_dir,
    "domainnet_quickdraw": domainnet_base_dir,
    "domainnet_painting": domainnet_base_dir,
}

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
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support" 
    "test_label_distribution": "iid", # Either "iid", "non_iid" "support"
    "test_samples_per_user": 200, # Only for non_iid test distribution
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

non_iid_sys_config = {
    "seed": 1,
    "num_users": 12,
    "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "load_existing": False,
    "device_ids": get_device_ids(
        num_users=12, num_client_per_gpu=7, available_gpus=[1, 2]
    ),
    "train_label_distribution": "non_iid",  # Either "iid", "non_iid" "support",
    "test_label_distribution": "non_iid",  # Either "iid" "support",
    "samples_per_user": 32,
    "test_samples_per_user": 32,
}

current_config = non_iid_sys_config