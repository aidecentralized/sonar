"""
This module runs collaborative learning experiments using the Scheduler class.
"""

import argparse
import logging
import subprocess

from utils.config_utils import load_config

logging.getLogger("PIL").setLevel(logging.INFO)

S_DEFAULT = "./configs/sys_config.py"
RANK_DEFAULT = 0

parser = argparse.ArgumentParser(description="Run collaborative learning experiments")
parser.add_argument(
    "-s",
    nargs="?",
    default=S_DEFAULT,
    type=str,
    help=f"filepath for system config, default: {S_DEFAULT}",
)

args = parser.parse_args()

sys_config = load_config(args.s)
print("Sys config loaded")

# 1. find the number of users in the system configuration
# 2. start separate processes by running python main.py for each user

num_users = sys_config["num_users"] + 1 # +1 for the super-node
for i in range(num_users):
    print(f"Starting process for user {i}")
    # start a Popen process
    subprocess.Popen(["python", "main.py", "-r", str(i)])