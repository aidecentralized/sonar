"""
This module runs main.py n times.
Usage: python main_grpc.py -n <number of nodes>
"""

import argparse
import subprocess
from typing import List

# Parse args
parser : argparse.ArgumentParser = argparse.ArgumentParser(description="Number of nodes to run on this machine")
parser.add_argument(
    "-n",
    nargs="?",
    type=int,
    help=f"number of nodes to run on this machine",
)

parser.add_argument(
    "-host",
    nargs="?",
    type=str,
    help=f"host address of the nodes",
)

parser.add_argument(
    "-dev",
    nargs="?",
    type=bool,
    help=f"whether or not development testing",
)

args : argparse.Namespace = parser.parse_args()

# Command for opening each process
command_list: List[str] = ["python", "main.py", "-host", args.host]
if args.dev == True:
    command_list: List[str] = ["python", "main.py", "-b", "./configs/algo_config_test.py", "-s", "./configs/sys_config_test.py", "-host", args.host]

# Start process for each user
for i in range(args.n):
    print(f"Starting process for user {i}")
    # start a Popen process
    subprocess.Popen(command_list)
