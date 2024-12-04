from typing import Dict, List, Literal, Optional
import random
from utils.types import ConfigType

from .algo_config_test import (
    traditional_fl
)

def get_device_ids(num_users: int, gpus_available: List[int | Literal["cpu"]]) -> Dict[str, List[int | Literal["cpu"]]]:
    """
    Get the GPU device IDs for the users.
    """
    # TODO: Make it multi-host
    device_ids: Dict[str, List[int | Literal["cpu"]]] = {}
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
    seed: Optional[int] = 1,
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

        # List of node indices to assign
        node_indices = list(range(1, total_users + 1))
        # Seed for reproducibility
        random.seed(seed)
        # Shuffle the node indices based on the seed
        random.shuffle(node_indices)

        # Assign nodes based on the shuffled indices
        current_index = 0
        for algo_index, num_nodes in distribution.items():
            for i in range(num_nodes):
                node_id = node_indices[current_index]
                algo_config_map[f"node_{node_id}"] = algo_configs[algo_index]
                current_index += 1
    else:
        raise ValueError(f"Invalid assignment method: {assignment_method}")
    # print("algo config mapping is: ", algo_config_map)
    return algo_config_map

CIFAR10_DSET = "cifar10"
CIAR10_DPATH = "./datasets/imgs/cifar10/"

# DUMP_DIR = "../../../../../../../home/"
DUMP_DIR = "/tmp/"

NUM_COLLABORATORS = 1
num_users = 4

dropout_dict = {
    "distribution_dict": { # leave dict empty to disable dropout
        "method": "uniform",  # "uniform", "normal"
        "parameters": {} # "mean": 0.5, "std": 0.1 in case of normal distribution
    },
    "dropout_rate": 0.0, # cutoff for dropout: [0,1]
    "dropout_correlation": 0.0, # correlation between dropouts of successive rounds: [0,1]
}

dropout_dicts = {"node_0": {}}
for i in range(1, num_users + 1):
    dropout_dicts[f"node_{i}"] = dropout_dict

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
    "algos": get_algo_configs(num_users=num_users, algo_configs=[traditional_fl]),  # type: ignore
    # "samples_per_user": 50000 // num_users,  # distributed equally
    "samples_per_user": 100,
    "train_label_distribution": "non_iid",
    "test_label_distribution": "iid",
    "alpha_data": 1.0,
    "exp_keys": [],
    "dropout_dicts": dropout_dicts,
    "test_samples_per_user": 100,
    "log_memory": True,
    # "streaming_aggregation": True, # Make it true for fedstatic
    "assign_based_on_host": True,
    "hostname_to_device_ids": {
        "matlaber1": [2, 3, 4, 5, 6, 7],
        "matlaber12": [0, 1, 2, 3],
        "matlaber3": [0, 1, 2, 3],
        "matlaber4": [0, 2, 3, 4, 5, 6, 7],
    },
    "workflow_test": True,
}
current_config = grpc_system_config