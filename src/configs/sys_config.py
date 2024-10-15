# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
from typing import Dict, List, Literal, Optional
import random
from utils.types import ConfigType

# from utils.config_utils import get_sliding_window_support, get_device_ids
from .algo_config import (
    malicious_algo_config_list,
    default_config_list,
    fedstatic,
)

sliding_window_8c_4cpc_support = {
    "1": [0, 1, 2, 3],
    "2": [1, 2, 3, 4],
    "3": [2, 3, 4, 5],
    "4": [3, 4, 5, 6],
    "5": [4, 5, 6, 7],
    "6": [5, 6, 7, 8],
    "7": [6, 7, 8, 9],
    "8": [7, 8, 9, 0],
}


def get_device_ids(num_users: int, gpus_available: List[int]) -> Dict[str, List[int]]:
    """
    Get the GPU device IDs for the users.
    """
    # TODO: Make it multi-host
    device_ids: Dict[str, List[int]] = {}
    for i in range(num_users + 1):  # +1 for the super-node
        index = i % len(gpus_available)
        gpu_id = gpus_available[index]
        device_ids[f"node_{i}"] = [gpu_id]
    return device_ids


def get_algo_configs(
    num_users: int,
    algo_configs: List[ConfigType],
    assignment_method: Literal[
        "sequential", "random", "mapping", "distribution"
    ] = "sequential",
    mapping: Optional[List[int]] = None,
    distribution: Optional[Dict[int, int]] = None,
) -> Dict[str, ConfigType]:
    """
    Assign an algorithm configuration to each node, allowing for repetition.
    sequential: Assigns the algo_configs sequentially to the nodes
    random: Assigns the algo_configs randomly to the nodes
    mapping: Assigns the algo_configs based on the mapping of node index to algo index provided
    distribution: Assigns the algo_configs based on the distribution of algo index to number of nodes provided
    """
    algo_config_map: Dict[str, ConfigType] = {}
    algo_config_map["node_0"] = algo_configs[0]  # Super-node
    if assignment_method == "sequential":
        for i in range(1, num_users + 1):
            algo_config_map[f"node_{i}"] = algo_configs[i % len(algo_configs)]
    elif assignment_method == "random":
        for i in range(1, num_users + 1):
            algo_config_map[f"node_{i}"] = random.choice(algo_configs)
    elif assignment_method == "mapping":
        if not mapping:
            raise ValueError("Mapping must be provided for assignment method 'mapping'")
        assert len(mapping) == num_users
        for i in range(1, num_users + 1):
            algo_config_map[f"node_{i}"] = algo_configs[mapping[i - 1]]
    elif assignment_method == "distribution":
        if not distribution:
            raise ValueError(
                "Distribution must be provided for assignment method 'distribution'"
            )
        total_users = sum(distribution.values())
        assert total_users == num_users
        current_index = 1
        for algo_index, num_nodes in distribution.items():
            for i in range(num_nodes):
                algo_config_map[f"node_{current_index}"] = algo_configs[algo_index]
                current_index += 1
    else:
        raise ValueError(f"Invalid assignment method: {assignment_method}")
    # print("algo config mapping is: ", algo_config_map)
    return algo_config_map


def get_domain_support(
    num_users: int, base: str, domains: List[int] | List[str]
) -> Dict[str, str]:
    assert num_users % len(domains) == 0

    users_per_domain = num_users // len(domains)
    support: Dict[str, str] = {}
    support["0"] = f"{base}_{domains[0]}"
    for i in range(1, num_users + 1):
        support[str(i)] = f"{base}_{domains[(i-1) // users_per_domain]}"
    return support


DOMAINNET_DMN = ["real", "sketch", "clipart"]


def get_domainnet_support(num_users: int, domains: List[str] = DOMAINNET_DMN):
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

CAMELYON17_DMN = [0, 3, 4]  # + 1, 2 in test set
CAMELYON17_DMN_EXT = [0, 1, 2, 3, 4]  # + 1, 2 in test set


def get_camelyon17_support(num_users: int, domains: List[int] = CAMELYON17_DMN):
    return get_domain_support(num_users, "wilds_camelyon17", domains)


DIGIT_FIVE_2 = ["svhn", "mnist_m"]
DIGIT_FIVE = ["svhn", "mnist_m", "synth_digits"]
DIGIT_FIVE_5 = ["mnist", "usps", "svhn", "mnist_m", "synth_digits"]


def get_digit_five_support(num_users: int, domains: List[str] = DIGIT_FIVE):
    return get_domain_support(num_users, "", domains)


digit_five_dpath = {
    "mnist": "./imgs/mnist",
    "usps": "./imgs/usps",
    "svhn": "./imgs/svhn",
    "mnist_m": "./imgs/MNIST-M",
    "synth_digits": "./imgs/syn_digit",
}

CIFAR10_DSET = "cifar10"
CIAR10_DPATH = "./datasets/imgs/cifar10/"

NUM_COLLABORATORS = 1
DUMP_DIR = "/mas/camera/Experiments/SONAR/abhi/"

mpi_system_config: ConfigType = {
    "exp_id": "",
    "comm": {"type": "MPI"},
    "num_collaborators": NUM_COLLABORATORS,
    "dset": CIFAR10_DSET,
    "dump_dir": DUMP_DIR,
    "dpath": CIAR10_DPATH,
    "seed": 32,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    #  "device_ids": {"node_0": [0], "node_1": [0], "node_2": [0], "node_3": [0]},
    "device_ids": get_device_ids(num_users=3, gpus_available=[1, 2]),
    # use this when the list needs to be imported from the algo_config
    # "algo": get_algo_configs(num_users=3, algo_configs=algo_configs_list),
    "algos": get_algo_configs(
        num_users=3,
        algo_configs=malicious_algo_config_list,
        assignment_method="distribution",
        distribution={0: 1, 1: 1, 2: 1},
    ),  # type: ignore
    "samples_per_user": 1000,  # TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",  # Either "iid", "non_iid" "support"
    "test_label_distribution": "iid",  # Either "iid", "non_iid" "support"
    "test_samples_per_user": 200,  # Only for non_iid test distribution
    "exp_keys": [],
}

mpi_non_iid_sys_config: ConfigType = {
    "exp_id": "",
    "comm": {"type": "MPI"},
    "seed": 1,
    "num_collaborators": NUM_COLLABORATORS,
    # "experiment_path": "./experiments/",
    "dset": CIFAR10_DSET,
    "dump_dir": DUMP_DIR,
    "dpath": CIAR10_DPATH,
    "load_existing": False,
    "device_ids": get_device_ids(num_users=3, gpus_available=[0, 3]),
    "algo": get_algo_configs(num_users=3, algo_configs=default_config_list),  # type: ignore
    "train_label_distribution": "non_iid",  # Either "iid", "non_iid" "support",
    "test_label_distribution": "non_iid",  # Either "iid" "support",
    "samples_per_user": 256,
    "test_samples_per_user": 100,
    "exp_keys": [],
}

L2C_users = 3
mpi_L2C_sys_config: ConfigType = {
    "exp_id": "",
    "comm": {"type": "MPI"},
    "seed": 1,
    "num_collaborators": NUM_COLLABORATORS,
    # "experiment_path": "./experiments/",
    "dset": CIFAR10_DSET,
    "dump_dir": DUMP_DIR,
    "dpath": CIAR10_DPATH,
    "load_existing": False,
    "device_ids": get_device_ids(num_users=3, gpus_available=[1, 2]),
    "algo": get_algo_configs(num_users=3, algo_configs=default_config_list),  # type: ignore
    "train_label_distribution": "iid",  # Either "iid", "non_iid" "support",
    "test_label_distribution": "iid",  # Either "iid" "support",
    "samples_per_user": 32,
    "test_samples_per_user": 32,
    "validation_prop": 0.05,
    "exp_keys": [],
}

mpi_metaL2C_support_sys_config: ConfigType = {
    "exp_id": "",
    "comm": {"type": "MPI"},
    "seed": 1,
    "num_collaborators": NUM_COLLABORATORS,
    # "experiment_path": "./experiments/",
    "dset": CIFAR10_DSET,
    "dump_dir": DUMP_DIR,
    "dpath": CIAR10_DPATH,
    "load_existing": False,
    "device_ids": get_device_ids(num_users=3, gpus_available=[1, 2]),
    "algo": get_algo_configs(num_users=3, algo_configs=default_config_list),  # type: ignore
    "train_label_distribution": "support",  # Either "iid", "non_iid" "support",
    "test_label_distribution": "support",  # Either "iid" "support",
    "support": sliding_window_8c_4cpc_support,
    "samples_per_user": 32,
    "test_samples_per_user": 32,
    "validation_prop": 0.05,
    "exp_keys": [],
}

mpi_digitfive_sys_config: ConfigType = {
    "exp_id": "",
    "comm": {"type": "MPI"},
    "seed": 1,
    "num_collaborators": NUM_COLLABORATORS,
    "load_existing": False,
    "dump_dir": DUMP_DIR,
    "device_ids": get_device_ids(num_users=3, gpus_available=[6, 7]),
    "algo": get_algo_configs(num_users=3, algo_configs=default_config_list),  # type: ignore
    # Dataset params
    "dset": get_digit_five_support(
        3
    ),  # get_camelyon17_support(fedcentral_client), #get_domainnet_support(fedcentral_client),
    "dpath": digit_five_dpath,  # wilds_dpath,#domainnet_dpath,
    "train_label_distribution": "iid",  # Either "iid", "shard" "support",
    "test_label_distribution": "iid",  # Either "iid" "support",
    "samples_per_user": 256,
    "test_samples_per_class": 100,
    "community_type": "dataset",
    "exp_keys": [],
}

swarm_users = 3
mpi_domainnet_sys_config: ConfigType = {
    "exp_id": "",
    "comm": {"type": "MPI"},
    "seed": 1,
    "num_collaborators": NUM_COLLABORATORS,
    "load_existing": False,
    "dump_dir": DUMP_DIR,
    "device_ids": get_device_ids(num_users=swarm_users, gpus_available=[3, 4]),
    "algo": get_algo_configs(num_users=swarm_users, algo_configs=default_config_list),  # type: ignore
    # Dataset params
    "dset": get_domainnet_support(
        swarm_users
    ),  # get_camelyon17_support(fedcentral_client), #get_domainnet_support(fedcentral_client),
    "dpath": domainnet_dpath,  # wilds_dpath,#domainnet_dpath,
    "train_label_distribution": "iid",  # Either "iid", "shard" "support",
    "test_label_distribution": "iid",  # Either "iid" "support",
    "samples_per_user": 32,
    "test_samples_per_class": 100,
    "community_type": "dataset",
    "exp_keys": [],
}

object_detect_system_config: ConfigType = {
    "exp_id": "",
    "num_users": 1,
    "num_collaborators": NUM_COLLABORATORS,
    "experiment_path": "./experiments/",
    "dset": "pascal",
    "dump_dir": DUMP_DIR,
    "dpath": "./datasets/pascal/VOCdevkit/VOC2012/",
    "seed": 37,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [1], "node_1": [2]},
    "algo": get_algo_configs(num_users=2, algo_configs=default_config_list),  # type: ignore
    "samples_per_user": 100,  # TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "exp_keys": [],
}

num_users = 9
gpu_ids = [2, 3, 5, 6]
grpc_system_config: ConfigType = {
    "exp_id": "static",
    "num_users": num_users,
    "num_collaborators": NUM_COLLABORATORS,
    "comm": {"type": "GRPC", "synchronous": True, "peer_ids": ["localhost:50048"]},  # The super-node
    "dset": CIFAR10_DSET,
    "dump_dir": DUMP_DIR,
    "dpath": CIAR10_DPATH,
    "seed": 2,
    "device_ids": get_device_ids(num_users, gpu_ids),
    # "algos": get_algo_configs(num_users=num_users, algo_configs=default_config_list),  # type: ignore
    "algos": get_algo_configs(num_users=num_users, algo_configs=[fedstatic]),  # type: ignore
    "samples_per_user": 50000 // num_users,  # distributed equally
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "exp_keys": [],
}

current_config = grpc_system_config
# current_config = mpi_system_config
