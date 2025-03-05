"""
Given a set of experiment keys to run,
this module writes the config files for each experiment key
and runs the main.py script for each experiment
"""

import argparse
from copy import deepcopy
from pprint import pprint
import subprocess
from typing import Dict, List

from utils.types import ConfigType
from utils.config_utils import process_config
from utils.post_hoc_plot_utils import aggregate_metrics_across_users, plot_all_metrics # type: ignore

from configs.sys_config import get_algo_configs, get_device_ids

post_hoc_plot: bool = False

# for each experiment key, write the modifications to the config file
topologies: Dict[str, Dict[str, int|float|str]] = {
    "ring": {},
    "torus": {},
    "erdos_renyi": {"p": 0.1},
    "watts_strogatz": {"k": 4, "p": 0.1},
    "fully_connected": {},
}

SOURCE_MACHINE = 'm4'

ROUNDS = 200
MODEL = "resnet18"
BATCH_SIZE = 256
LR = 3e-4
NUM_USERS = 36
NUM_COLLABORATORS = 1
SAMPLES_PER_USER = 96

CIFAR10_DSET = "cifar10"
CIAR10_DPATH = "./datasets/imgs/cifar10/"

DUMP_DIR = "/mas/camera/Experiments/SONAR/abhi/"

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

dropout_dict = { # type: ignore
    "distribution_dict": { # leave dict empty to disable dropout
        "method": "uniform",  # "uniform", "normal"
        "parameters": {} # "mean": 0.5, "std": 0.1 in case of normal distribution
    },
    "dropout_rate": 0.0, # cutoff for dropout: [0,1]
    "dropout_correlation": 0.0, # correlation between dropouts of successive rounds: [0,1]
}

dropout_dicts = {"node_0": {}} # type: ignore
for i in range(1, NUM_USERS + 1):
    dropout_dicts[f"node_{i}"] = dropout_dict

# for swift or fedavgpush, just modify the algo_configs list
# for swift, synchronous should preferable be False
gpu_ids = [1, 2, 3, 4, 5, 6, 7]
grpc_system_config: ConfigType = {
    "exp_id": f"static_alpha_{SOURCE_MACHINE}",
    "num_users": NUM_USERS,
    "num_collaborators": NUM_COLLABORATORS,
    "comm": {"type": "GRPC", "synchronous": True, "peer_ids": ["localhost:50048"]},  # type: ignore
    "dset": get_domainnet_support(NUM_USERS),
    "dump_dir": DUMP_DIR,
    "dpath": domainnet_dpath,
    "seed": 2,
    "device_ids": get_device_ids(NUM_USERS, gpu_ids),
    "samples_per_user": SAMPLES_PER_USER,
    "train_label_distribution": "iid",
    "test_label_distribution": "iid", # for domainnet it will not matter
    "exp_keys": [],
    "dropout_dicts": dropout_dicts,
    "log_memory": False,
}

fedstatic: ConfigType = {
    # Collaboration setup
    "algo": "fedstatic",
    # this will be populated by the loop below
    "topology": {}, # type: ignore
    "rounds": ROUNDS,

    # Model parameters
    "model": MODEL,
    "model_lr": LR,
    "batch_size": 256,
}

exp_dict: Dict[str, ConfigType] = {}

for topo in topologies.keys():
    sys_config = deepcopy(grpc_system_config)
    algo_config = deepcopy(fedstatic)
    algo_config["topology"] = {"name": topo}
    algo_config["topology"].update(topologies[topo]) # type: ignore
    print(f"Algo config for {topo}: {algo_config}")
    exp_dict[f"convergence_{topo}"] = { # type: ignore
        "algo_config": algo_config,
        "sys_config": sys_config,
    }

# add async fully connected (gossip) experiment
algo_config = deepcopy(fedstatic)
algo_config["topology"] = {"name": "fully_connected"}
sys_config = deepcopy(grpc_system_config)
sys_config["comm"]["synchronous"] = False # type: ignore
exp_dict["convergence_fully_connected_async"] = { # type: ignore
    "algo_config": algo_config,
    "sys_config": sys_config,
}

# add the traditional federated learning experiment
algo_config = deepcopy(fedstatic)
sys_config = deepcopy(grpc_system_config)
# delete the topology key
del algo_config["topology"]
# change the algo key
algo_config["algo"] = "fedavg"

exp_dict["convergence_traditional_fl"] = { # type: ignore
    "algo_config": algo_config,
    "sys_config": sys_config,
}

# parse the arguments
parser = argparse.ArgumentParser(description="host address of the nodes")
parser.add_argument(
    "-host",
    nargs="?",
    type=str,
    default="localhost",
    help="host address of the nodes",
)

args: argparse.Namespace = parser.parse_args()

print("\n", "*"*20, "Running the following experiments in this order:", "*"*20, "\n")
print(exp_dict.keys())

for exp_id, exp_config in exp_dict.items():
    # update the algo config with config settings
    base_algo_config: ConfigType = exp_config["algo_config"].copy() # type: ignore
    # update the sys config with config settings
    base_sys_config = exp_config["sys_config"].copy() # type: ignore

    # set up the full config file by combining the algo and sys config
    n: int = base_sys_config["num_users"] # type: ignore
    seed: int = base_sys_config["seed"] # type: ignore
    base_sys_config["algos"] = get_algo_configs(num_users=n, algo_configs=[base_algo_config], seed=seed) # type: ignore
    base_sys_config["device_ids"] = get_device_ids(n, gpu_ids) # type: ignore

    full_config = base_sys_config.copy() # type: ignore
    full_config["exp_id"] = exp_id # type: ignore

    # change print color to green
    print("\033[92m")
    print("\n" + "_" * 100, f"\nRunning experiment {exp_id} with the following config:\n")
    pprint(full_config) # type: ignore
    # reset print color
    print("\033[0m")

    config_filename = f"./configs/{SOURCE_MACHINE}_config.py"

    with open(config_filename, "w") as f:
        f.write("current_config = ")
        f.write(str(full_config)) # type: ignore

    # # start the supernode
    supernode_command: List[str] = ["python", "main.py", "-host", args.host, "-super", "true", "-s", config_filename]
    process = subprocess.Popen(supernode_command)

    # # start the nodes
    command_list: List[str] = ["python", "main.py", "-host", args.host, "-s", config_filename]
    for i in range(n):
        print(f"Starting process for user {i} exp {exp_id}")
        # start a Popen process
        subprocess.Popen(command_list)

    # # once the experiment is done, run the next experiment
    # # Wait for the supernode process to finish
    process.wait()

    # run the post-hoc analysis
    if post_hoc_plot:
        full_config = process_config(full_config) # type: ignore this populates the results path
        logs_dir = full_config["results_path"] + '/logs/'

        # aggregate metrics across all users
        aggregate_metrics_across_users(logs_dir)
        # plot all metrics
        plot_all_metrics(logs_dir)

    # Continue with the next set of commands after supernode finishes
    print(f"Supernode process {exp_id} finished. Proceeding to next set of commands.")