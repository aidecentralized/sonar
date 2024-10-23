"""
Given a set of experiment keys to run,
this module writes the config files for each experiment key
and runs the main.py script for each experiment
"""

import argparse
import subprocess
from typing import List

from configs.sys_config import grpc_system_config

# parse the arguments
parser = argparse.ArgumentParser(description="host address of the nodes")
parser.add_argument(
    "-host",
    nargs="?",
    type=str,
    help=f"host address of the nodes",
)

args = parser.parse_args()

# for each experiment key
# write the new config file
exp_ids = ["test_automation_1", "test_automation_2", "test_automation_3"]

for e, exp_id in enumerate(exp_ids):
    current_config = grpc_system_config
    current_config["exp_id"] = exp_id

    # write the config file as python file configs/temp_config.py
    with open("./configs/temp_config.py", "w") as f:
        f.write("current_config = ")
        f.write(str(current_config))

    n: int = current_config["num_users"]

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

    # Continue with the next set of commands after supernode finishes
    print(f"Supernode process {exp_id} finished. Proceeding to next set of commands.")