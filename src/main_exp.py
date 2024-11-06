"""
Given a set of experiment keys to run,
this module writes the config files for each experiment key
and runs the main.py script for each experiment
"""

import argparse
import subprocess
from typing import List

from utils.types import ConfigType
from utils.config_utils import process_config
from utils.post_hoc_plot_utils_copy import aggregate_metrics_across_users, plot_all_metrics

from configs.sys_config import get_algo_configs, get_device_ids
from configs.algo_config import traditional_fl
from configs.sys_config import grpc_system_config

post_hoc_plot: bool = True

# for each experiment key, write the modifications to the config file
gpu_ids = [2, 3, 5, 6]
exp_dict = {
    "experiment_1": {
        "algo_config": traditional_fl,
        "sys_config": grpc_system_config,
        "algo": {
            "rounds": 3,
        },
        "sys": {
            "seed": 3,
            "num_users": 3,
        },
    },
    "experiment_2": {
        "algo_config": traditional_fl,
        "sys_config": grpc_system_config,
        "algo": {
            "rounds": 4,
        },
        "sys": {
            "seed": 4,
            "num_users": 4,
        },
    },
}

# parse the arguments
parser = argparse.ArgumentParser(description="host address of the nodes")
parser.add_argument(
    "-host",
    nargs="?",
    type=str,
    help=f"host address of the nodes",
)

args = parser.parse_args()

for exp_id, exp_config in exp_dict.items():
    # update the algo config with config settings
    base_algo_config = exp_config["algo_config"].copy()
    base_algo_config.update(exp_config["algo"])

    # update the sys config with config settings
    base_sys_config = exp_config["sys_config"].copy()
    base_sys_config.update(exp_config["sys"])

    # set up the full config file by combining the algo and sys config
    n: int = base_sys_config["num_users"]
    seed: int = base_sys_config["seed"]
    base_sys_config["algos"] = get_algo_configs(num_users=n, algo_configs=[base_algo_config], seed=seed)
    base_sys_config["device_ids"] = get_device_ids(n, gpu_ids)

    full_config = base_sys_config.copy()
    full_config["exp_id"] = exp_id

    # write the config file as python file configs/temp_config.py
    with open("./configs/temp_config.py", "w") as f:
        f.write("current_config = ")
        f.write(str(full_config))

    # start the supernode
    supernode_command: List[str] = ["python", "main.py", "-host", args.host, "-super", "true", "-s", "./configs/temp_config.py"]
    process = subprocess.Popen(supernode_command)

    # start the nodes
    command_list: List[str] = ["python", "main.py", "-host", args.host, "-s", "./configs/temp_config.py"]
    for i in range(n):
        print(f"Starting process for user {i} exp {exp_id}")
        # start a Popen process
        subprocess.Popen(command_list)

    # once the experiment is done, run the next experiment
    # Wait for the supernode process to finish
    process.wait()

    # run the post-hoc analysis
    if post_hoc_plot:
        full_config = process_config(full_config) # this populates the results path
        logs_dir = full_config["results_path"] + '/logs/'

        # aggregate metrics across all users
        aggregate_metrics_across_users(logs_dir)
        # plot all metrics
        plot_all_metrics(logs_dir)

    # Continue with the next set of commands after supernode finishes
    print(f"Supernode process {exp_id} finished. Proceeding to next set of commands.")