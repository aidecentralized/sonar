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
from utils.post_hoc_plot_utils import aggregate_metrics_across_users, plot_all_metrics

from configs.sys_config import get_algo_configs, get_device_ids
from configs.algo_config import fedstatic
from configs.malicious_config import malicious_config_list
from configs.sys_config import grpc_system_config
import socket
import time

# Get the hostname
hostname = socket.gethostname()
superhost_name = "" # Fill in the superhost name
full_hostname = "" # Fill in the full hostname

post_hoc_plot: bool = True

algo_to_algo_index = {
    "data_poisoning": 0,
    "gradient_attack": 1,
    "backdoor_attack": 2,
    "bad_weights": 3,
    "sign_flip": 4,
    "label_flip": 5,
}

# for each experiment key, write the modifications to the config file
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
exp_dict = {}
num_nodes = 36
for num_collaborators in [num_nodes, 1]:
    for algo_name, algo_index in algo_to_algo_index.items():
        for topo in ["ring", "torus", "fully_connected", "erdos_renyi"]:
            for m in [0, 1, 4]:
                topo_config = {"name": topo}
                if topo == "erdos_renyi":
                    topo_config["p"] = 0.13
                exp_dict[f"topo_{topo}x{algo_name}_{m}_malicious_{num_collaborators}_colab_3_5"] = {
                    "algo_config": fedstatic,
                    "sys_config": grpc_system_config,
                    "malicious_config": malicious_config_list[algo_name],
                    "num_malicious": m,
                    "algo": {
                        "topology": topo_config,
                    },
                    "sys": {
                        "comm": {"type": "GRPC", "synchronous": True, "peer_ids": ["matlaber1.media.mit.edu:1112"]},
                        "num_users": num_nodes,
                        "num_collaborators": num_collaborators,
                        "samples_per_user": 50000 // num_nodes,
                        "seed": 2,
                        "assign_based_on_host": True,
                    },
                }

# parse the arguments
parser = argparse.ArgumentParser(description="host address of the nodes")

args = parser.parse_args()

skip = True
for exp_id, exp_config in exp_dict.items():
    if skip:
        skip = False
        continue
    print(f"Running experiment {exp_config}")
    # update the algo config with config settings
    base_algo_config = exp_config["algo_config"].copy()
    base_algo_config.update(exp_config["algo"])

    # update the sys config with config settings
    base_sys_config = exp_config["sys_config"].copy()
    base_sys_config.update(exp_config["sys"])

    # update the malicious config with config settings
    base_malicious_config = exp_config["malicious_config"].copy()
    base_malicious_config.update(base_algo_config)

    # set up the full config file by combining the algo and sys config
    n: int = base_sys_config["num_users"]
    seed: int = base_sys_config["seed"]
    m: int = exp_config["num_malicious"]
    base_sys_config["algos"] = get_algo_configs(num_users=n, algo_configs=[base_algo_config, base_malicious_config], seed=seed, assignment_method="distribution", distribution={0: n-m, 1: m})
    base_sys_config["device_ids"] = get_device_ids(n, gpu_ids)

    full_config = base_sys_config.copy()
    full_config["exp_id"] = exp_id

    # write the config file as python file configs/temp_config.py
    temp_config_path = "./configs/temp_config.py"
    with open(temp_config_path, "w") as f:
        f.write("current_config = ")
        f.write(str(full_config))

    superprocess = None
    all_processes = []

    # start the supernode
    if hostname == superhost_name:
        print("Starting supernode")
        supernode_command: List[str] = ["python", "main.py", "-host", full_hostname, "-super", "true", "-s", temp_config_path]
        superprocess = subprocess.Popen(supernode_command)
    else:
        print("Waiting for supernode to start")
    time.sleep(10)

    # start the nodes
    command_list: List[str] = ["python", "main.py", "-host", full_hostname, "-s", temp_config_path]
    for i in range(num_nodes):
        print(f"Starting process for user {i} exp {exp_id}")
        # start a Popen process
        all_processes.append(subprocess.Popen(command_list))

    # once the experiment is done, run the next experiment
    # Wait for the supernode process to finish
    if superprocess:
        superprocess.wait()
    else:
        # wait for all the processes to finish
        for process in all_processes:
            process.wait()
        # wait for 5 more minutes
        print("Processes done, waiting for 5 minutes")
        time.sleep(300)

    # run the post-hoc analysis
    if post_hoc_plot and superprocess is not None:
        full_config = process_config(full_config) # this populates the results path
        logs_dir = full_config["results_path"] + '/logs/'

        # aggregate metrics across all users
        aggregate_metrics_across_users(logs_dir)
        # plot all metrics
        plot_all_metrics(logs_dir)

    # Continue with the next set of commands after supernode finishes
    print(f"Supernode process {exp_id} finished. Proceeding to next set of commands.")