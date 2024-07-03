import argparse
from scheduler import Scheduler
import gc
import torch
import copy
import logging

logging.getLogger("PIL").setLevel(logging.INFO)

b_default = "./configs/algo_config.py"
s_default = "./configs/sys_config.py"

parser = argparse.ArgumentParser(description="Run collaborative learning experiments")
parser.add_argument(
    "-b",
    nargs="?",
    default=b_default,
    type=str,
    help="filepath for benchmark config, default: {}".format(b_default),
)
parser.add_argument(
    "-s",
    nargs="?",
    default=s_default,
    type=str,
    help="filepath for system config, default: {}".format(s_default),
)

args = parser.parse_args()

scheduler = Scheduler()
scheduler.assign_config_by_path(args.s, args.b)
print("Config loaded")


scheduler.install_config()
scheduler.initialize()
scheduler.run_job()