"""
This module runs collaborative learning experiments using the Scheduler class.
"""

import argparse
import logging

from scheduler import Scheduler

logging.getLogger("PIL").setLevel(logging.INFO)

B_DEFAULT = "./configs/algo_config.py"
S_DEFAULT = "./configs/sys_config.py"

parser = argparse.ArgumentParser(description="Run collaborative learning experiments")
parser.add_argument(
    "-b",
    nargs="?",
    default=B_DEFAULT,
    type=str,
    help=f"filepath for benchmark config, default: {B_DEFAULT}",
)
parser.add_argument(
    "-s",
    nargs="?",
    default=S_DEFAULT,
    type=str,
    help=f"filepath for system config, default: {S_DEFAULT}",
)
parser.add_argument(
    "-super",
    nargs="?",
    type=bool,
    help=f"whether to run the super node",
)
parser.add_argument(
    "-host",
    nargs="?",
    type=str,
    help=f"host address of the nodes",
)

args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args.s, args.b, args.super, args.host)
print("Config loaded")

scheduler.install_config()
scheduler.initialize()
scheduler.run_job()
