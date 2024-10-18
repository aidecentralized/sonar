"""
This module runs collaborative learning experiments using the Scheduler class.
"""

import argparse
import logging

from scheduler import Scheduler

logging.getLogger("PIL").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)  # Enable detailed logging

B_DEFAULT: str = "./configs/algo_config.py"
S_DEFAULT: str = "./configs/sys_config.py"

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
    help="whether to run the super node",
)
parser.add_argument(
    "-host",
    nargs="?",
    type=str,
    help="host address of the nodes",
)

args: argparse.Namespace = parser.parse_args()

scheduler: Scheduler = Scheduler()

# Assign the configuration from the file paths provided via arguments
scheduler.assign_config_by_path(args.s, args.b, args.super, args.host)
print("Config loaded")

# Log and check key configuration values to prevent errors like division by zero
num_users = scheduler.config.get("num_users", None)
if num_users is None:
    logging.error(
        "The number of users (num_users) is not defined in the configuration."
    )
    raise ValueError("num_users must be defined in the configuration.")
if num_users == 0:
    logging.error(
        "The number of users is set to 0, which will cause a ZeroDivisionError."
    )
    raise ValueError("num_users cannot be zero. Please check the configuration.")

logging.info(f"Running experiment with {num_users} users.")

scheduler.install_config()
scheduler.initialize()

# Run the job
scheduler.run_job()
