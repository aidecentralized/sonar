# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
from typing import Dict, List, Literal, Optional
import random
from utils.types import ConfigType

# from utils.config_utils import get_sliding_window_support, get_device_ids
from .algo_config import (
    default_config_list,
    fedstatic,
    traditional_fl,
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

NUM_COLLABORATORS = 1
DUMP_DIR = "/mas/camera/Experiments/SONAR/jyuan/"

num_users = 10
dropout_dict = {}
dropout_dicts = {"node_0": {}}
for i in range(1, num_users + 1):
    dropout_dicts[f"node_{i}"] = dropout_dict

# for swift or fedavgpush, just modify the algo_configs list
# for swift, synchronous should preferable be False
gpu_ids = [2, 3, 7]
rtc_config: ConfigType = {
    "exp_id": "test_train_10_clients3",
    "num_users": num_users,
    "session_id": 1111,
    "num_collaborators": NUM_COLLABORATORS,
    "comm": {"type": "RTC"},
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
    "dropout_dicts": dropout_dicts,
    "exp_keys": [],
}

current_config = rtc_config
