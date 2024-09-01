# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
from typing import Dict, List
# from utils.config_utils import get_sliding_window_support, get_device_ids


def get_device_ids(num_users: int, gpus_available: List[int]) -> Dict[str, List[int]]:
    """
    Get the GPU device IDs for the users.
    """
    # TODO: Make it multi-host
    device_ids: Dict[str, List[int]] = {}
    for i in range(num_users + 1): # +1 for the super-node
        index = i % len(gpus_available)
        gpu_id = gpus_available[index]
        device_ids[f"node_{i}"] = [gpu_id]
    return device_ids

def get_domain_support(num_users: int, base: str, domains: List[int]|List[str]) -> Dict[str, str]:
    assert num_users % len(domains) == 0

    users_per_domain = num_users // len(domains)
    support: Dict[str, str] = {}
    support["0"] = f"{base}_{domains[0]}"
    for i in range(1, num_users + 1):
        support[str(i)] = f"{base}_{domains[(i-1) // users_per_domain]}"
    return support

DOMAINNET_DMN = ["real", "sketch", "clipart"]

def get_domainnet_support(num_users: int, domains: List[str]=DOMAINNET_DMN):
    return get_domain_support(num_users, "domainnet", domains)

domainnet_base_dir = "/u/abhi24/matlaberp2/p2p/imgs/domainnet/"
domainnet_dpath = {
    "domainnet_real": domainnet_base_dir,
    "domainnet_sketch": domainnet_base_dir,
    "domainnet_clipart": domainnet_base_dir,
    "domainnet_infograph": domainnet_base_dir,
    "domainnet_quickdraw": domainnet_base_dir,
    "domainnet_painting": domainnet_base_dir,
}

CAMELYON17_DMN = [0, 3, 4] # + 1, 2 in test set
CAMELYON17_DMN_EXT = [0,1, 2, 3, 4] # + 1, 2 in test set

def get_camelyon17_support(num_users: int, domains: List[int]=CAMELYON17_DMN):
    return get_domain_support(num_users, "wilds_camelyon17", domains)

DIGIT_FIVE_2 = ["svhn", "mnist_m"] 
DIGIT_FIVE = ["svhn", "mnist_m", "synth_digits"] 
DIGIT_FIVE_5 = ["mnist", "usps", "svhn", "mnist_m", "synth_digits"] 

def get_digit_five_support(num_users:int, domains:List[str]=DIGIT_FIVE):
    return get_domain_support(num_users, "", domains)

digit_five_dpath = {
    "mnist": "./imgs/mnist",
    "usps": "./imgs/usps",
    "svhn": "./imgs/svhn",
    "mnist_m": "./imgs/MNIST-M",
    "synth_digits": "./imgs/syn_digit"
}

mpi_system_config = {
    "comm": {
        "type": "MPI"
    },
    "num_users": 4,
    # "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 32,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [0], "node_1": [0],"node_2": [0], "node_3": [0]},
    "samples_per_user": 1000, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support" 
    "test_label_distribution": "iid", # Either "iid", "non_iid" "support"
    "test_samples_per_user": 200, # Only for non_iid test distribution
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

mpi_non_iid_sys_config = {
    "comm": {
        "type": "MPI"
    },
    "seed": 1,
    "num_users": 3,
    # "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "load_existing": False,
    "device_ids": get_device_ids(num_users=3, gpus_available=[0, 3]),
    "train_label_distribution": "non_iid",  # Either "iid", "non_iid" "support",
    "test_label_distribution": "non_iid",  # Either "iid" "support",
    "samples_per_user": 256,
    "test_samples_per_user": 100,
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

L2C_users = 3
mpi_L2C_sys_config = {
    "comm": {
        "type": "MPI"
    },
    "seed": 1,
    "num_users": 3,
    # "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "load_existing": False,
    "device_ids": get_device_ids(num_users=3, gpus_available=[1, 2]),
    "train_label_distribution": "iid",  # Either "iid", "non_iid" "support",
    "test_label_distribution": "iid",  # Either "iid" "support",
    "samples_per_user": 32,
    "test_samples_per_user": 32,
    "validation_prop": 0.05,
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

mpi_digitfive_sys_config = {
    "comm": {
        "type": "MPI"
    },
    "seed": 1,
    "num_users": 3,

    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_users=3, gpus_available=[6, 7]),

    # Dataset params 
    "dset": get_digit_five_support(3),#get_camelyon17_support(fedcentral_client), #get_domainnet_support(fedcentral_client),
    "dpath": digit_five_dpath, #wilds_dpath,#domainnet_dpath,
    "train_label_distribution": "iid", # Either "iid", "shard" "support", 
    "test_label_distribution": "iid", # Either "iid" "support",     
    "samples_per_user": 256,
    "test_samples_per_class": 100,
    "community_type": "dataset",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

swarm_users = 3
mpi_domainnet_sys_config = {
    "comm": {
        "type": "MPI"
    },
    "seed": 1,
    "num_users": 3,

    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_users=swarm_users, gpus_available=[3, 4]),

    # Dataset params 
    "dset": get_domainnet_support(swarm_users),#get_camelyon17_support(fedcentral_client), #get_domainnet_support(fedcentral_client),
    "dpath": domainnet_dpath, #wilds_dpath,#domainnet_dpath,
    "train_label_distribution": "iid", # Either "iid", "shard" "support", 
    "test_label_distribution": "iid", # Either "iid" "support",     
    "samples_per_user": 32,
    "test_samples_per_class": 100,
    "community_type": "dataset",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

object_detect_system_config = {
    "num_users": 1,
    "experiment_path": "./experiments/",
    "dset": "pascal",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/pascal/VOCdevkit/VOC2012/",
    "seed": 37,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [1], "node_1": [2]},
    "samples_per_user": 100, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

num_users = 20
gpu_ids = [1, 2, 3, 4, 5, 6, 7]
grpc_system_config = {
    "num_users": num_users,
    "comm": {
        "type": "GRPC",
        "peer_ids": ["localhost:50050"] # The super-node
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 2,
    "device_ids": get_device_ids(num_users, gpu_ids),
    "samples_per_user": 50000 // num_users, # distributed equally
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

current_config = grpc_system_config
