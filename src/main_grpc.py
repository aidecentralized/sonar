"""
This module runs main.py n times.
Usage: python main_grpc.py -n <number of nodes>
"""

import argparse
import subprocess
from typing import List

parser = argparse.ArgumentParser(description="Number of nodes to run on this machine")
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

args = parser.parse_args()

command_list: List[str] = ["python", "main.py", "-host", args.host]
# if the super-node is to be started on this machine

for i in range(args.n):
    print(f"Starting process for user {i}")
    # start a Popen process
    subprocess.Popen(command_list)
