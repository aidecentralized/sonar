folder = "expt_dump/"
exp_folder = "name_of_experiment_folder"
stats = [
    {
        "file": "test_acc_per_client_per_round.npy",
        "name": "test accuracy",
        "order": "max",
        "round_step": 1,
    },
    {
        "file": "tr_acc_per_client_per_round.npy",
        "name": "train accuracy",
        "order": "max",
        "round_step": 1,
    },
    {
        "file": "tr_loss_per_client_per_round.npy",
        "name": "train loss",
        "order": "min",
        "round_step": 1,
    },
]

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from utils.log_utils import LogUtils
import numpy as np

path_to_log = folder + exp_folder + "/logs/"
config = {"log_path": path_to_log, "load_existing": True}
log_utils = LogUtils(config)

for stat in stats:
    file = stat["file"]
    name = stat["name"]
    order = stat["order"]
    round_step = stat["round_step"]

    stats_per_client = np.load(f"{path_to_log}npy/{file}")
    if order == "max":
        log_utils.log_max_stats_per_client(stats_per_client, round_step, name)
    elif order == "min":
        log_utils.log_min_stats_per_client(stats_per_client, round_step, name)
    else:
        raise ValueError("Order should be max or min")
