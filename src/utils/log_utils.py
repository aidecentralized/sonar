"""
This module provides utility functions and classes for handling logging,
copying source code, and normalizing images in a distributed learning setting.
"""

import os
import logging
import sys
from glob import glob
from shutil import copytree, copy2
from typing import Any, Dict
import torch
import torchvision.transforms as T  # type: ignore
from torchvision.utils import make_grid, save_image  # type: ignore
from tensorboardX import SummaryWriter  # type: ignore
import numpy as np


def deprocess(img: torch.Tensor) -> torch.Tensor:
    """
    Deprocesses an image tensor by normalizing it to the original range.

    Args:
        img (torch.Tensor): Image tensor to deprocess.

    Returns:
        torch.Tensor: Deprocessed image tensor.
    """
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    img = inv_normalize(img)
    img = 255 * img
    return img.type(torch.uint8)


def check_and_create_path(path: str):
    """
    Checks if the specified path exists and prompts the user for action if it does.
    Creates the directory if it does not exist.

    Args:
        path (str): Path to check and create if necessary.
    """
    if os.path.isdir(path):
        color_code = "\033[94m"  # Blue text
        reset_code = "\033[0m"  # Reset to default color
        print(f"{color_code}Experiment in {path} already present. Exiting.")
        print(f"Please do: rm -rf {path} to delete the folder.{reset_code}")
        sys.exit()
    else:
        os.makedirs(path)


def copy_source_code(config: Dict[str, Any]) -> None:
    """
    Copy source code to experiment folder for reproducibility.

    Args:
        config (dict): Configuration dictionary with the results path.
    """
    path = config["results_path"]
    print("exp path:", path)
    if config["load_existing"]:
        print("Continue with loading checkpoint")
        return
    check_and_create_path(path)
    denylist = [
        "./__pycache__/",
        "./.ipynb_checkpoints/",
        "./expt_dump/",
        "./helper_scripts/",
        "./datasets/",
        "./imgs/",
        "./expt_dump_old/",
        "./comparison_plots/",
        "./toy_exp/",
        "./toy_exp_ml/",
        "./toy_exp.py",
        "./toy_exp_ml.py",
        "/".join(path.split("/")[:-1]) + "/",
    ]
    folders = glob(r"./*/")
    print(denylist, folders)

    for file_ in glob(r"./*.py"):
        copy2(file_, path)
    for file_ in glob(r"./*.json"):
        copy2(file_, path)
    for folder in folders:
        if folder not in denylist:
            copytree(folder, path + folder[1:])
    os.mkdir(config["saved_models"])
    os.makedirs(config["log_path"], exist_ok=True)
    print("source code copied to exp_dump")


class LogUtils:
    """
    Utility class for logging and saving experiment data.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        log_dir = config["log_path"]
        load_existing = config["load_existing"]
        log_format = (
            "%(asctime)s::%(levelname)s::%(name)s::"
            "%(filename)s::%(lineno)d::%(message)s"
        )
        logging.basicConfig(
            filename=f"{log_dir}/log_console.log",
            level="DEBUG",
            format=log_format,
        )
        logging.getLogger().addHandler(logging.StreamHandler())
        self.log_dir = log_dir
        self.init_tb(load_existing)
        self.init_npy()
        self.init_summary()

    def init_summary(self):
        """
        Initialize summary file for logging.
        """
        self.summary_file = open(f"{self.log_dir}/summary.txt", "w", encoding="utf-8")

    def init_tb(self, load_existing: bool):
        """
        Initialize TensorBoard logging.

        Args:
            load_existing (bool): Whether to load existing logs.
        """
        tb_path = f"{self.log_dir}/tensorboard"
        if not load_existing:
            os.makedirs(tb_path, exist_ok=True)
        self.writer = SummaryWriter(tb_path)

    def init_npy(self):
        """
        Initialize directory for saving numpy arrays.
        """
        npy_path = f"{self.log_dir}/npy"
        if not os.path.exists(npy_path) or not os.path.isdir(npy_path):
            os.makedirs(npy_path)

    def log_summary(self, text: str):
        """
        Add summary text to the summary file for logging.
        """
        if self.summary_file:
            self.summary_file.write(text + "\n")
            self.summary_file.flush()
        else:
            raise ValueError(
                "Summary file is not initialized. Call init_summary() first."
            )

    def log_image(self, imgs: torch.Tensor, key: str, iteration: int):
        """
        Log image to file and TensorBoard.

        Args:
            imgs (torch.Tensor): Tensor of images to log.
            key (str): Key for the logged image.
            iteration (int): Current iteration number.
        """
        grid_img = make_grid(imgs.detach().cpu(), normalize=True, scale_each=True)
        save_image(grid_img, f"{self.log_dir}/{iteration}_{key}.png")
        self.writer.add_image(key, grid_img.numpy(), iteration)

    def log_console(self, msg: str):
        """
        Log a message to the console.

        Args:
            msg (str): Message to log.
        """
        logging.info(msg)

    def log_tb(self, key: str, value: float | int, iteration: int):
        """
        Log a scalar value to TensorBoard.

        Args:
            key (str): Key for the logged value.
            value (float): Value to log.
            iteration (int): Current iteration number.
        """
        self.writer.add_scalar(key, value, iteration)  # type: ignore

    def log_npy(self, key: str, value: np.ndarray):
        """
        Save a numpy array to file.

        Args:
            key (str): Key for the saved array.
            value (numpy.ndarray): Array to save.
        """
        np.save(f"{self.log_dir}/npy/{key}.npy", value)

    def log_max_stats_per_client(
        self, stats_per_client: np.ndarray, round_step: int, metric: str
    ):
        """
        Log maximum statistics per client.

        Args:
            stats_per_client (numpy.ndarray): Statistics for each client.
            round_step (int): Step size for rounds.
            metric (str): Metric being logged.
        """
        self.__log_stats_per_client__(stats_per_client, round_step, metric, is_max=True)

    def log_min_stats_per_client(self, stats_per_client, round_step, metric):
        """
        Log minimum statistics per client.

        Args:
            stats_per_client (numpy.ndarray): Statistics for each client.
            round_step (int): Step size for rounds.
            metric (str): Metric being logged.
        """
        self.__log_stats_per_client__(
            stats_per_client, round_step, metric, is_max=False
        )

    def __log_stats_per_client__(
        self, stats_per_client, round_step, metric, is_max=False
    ):
        """
        Internal method to log statistics per client.

        Args:
            stats_per_client (numpy.ndarray): Statistics for each client.
            round_step (int): Step size for rounds.
            metric (str): Metric being logged.
            is_max (bool): Whether to log maximum or minimum statistics.
        """
        if is_max:
            best_round_per_client = np.argmax(stats_per_client, axis=1) * round_step
            best_val_per_client = np.max(stats_per_client, axis=1)
        else:
            best_round_per_client = np.argmin(stats_per_client, axis=1) * round_step
            best_val_per_client = np.min(stats_per_client, axis=1)

        minmax = "max" if is_max else "min"
        self.summary_file.write(
            f"============== {minmax} {metric} per client ==============\n"
        )
        for client_idx, (best_round, best_val) in enumerate(
            zip(best_round_per_client, best_val_per_client)
        ):
            self.summary_file.write(
                f"Client {client_idx + 1} : {best_val} at round {best_round}\n"
            )
        self.summary_file.write(
            f"Mean of {minmax} {metric} : {np.mean(best_val_per_client)}, quantiles: {np.quantile(best_val_per_client, [0.25, 0.75])}\n"
        )

    def log_tb_round_stats(self, round_stats, stats_to_exclude, current_round):
        """
        Log round statistics to TensorBoard.

        Args:
            round_stats (list): List of round statistics for each client.
            stats_to_exclude (list): List of statistics keys to exclude from logging.
            current_round (int): Current round number.
        """
        stats_key = round_stats[0].keys()
        for key in stats_key:
            if key not in stats_to_exclude:
                average = 0
                for client_id, stats in enumerate(round_stats, 1):
                    self.log_tb(f"{key}/client{client_id}", stats[key], current_round)
                    average += stats[key]
                average /= len(round_stats)
                self.log_tb(f"{key}/clients", average, current_round)

    def log_experiments_stats(self, global_stats):
        """
        Log experiment statistics.

        Args:
            global_stats (dict): Dictionary of global statistics.
        """
        basic_stats = {
            "train_loss": "min",
            "train_acc": "max",
            "test_acc": "max",
            "test_acc_before_training": "max",
            "test_acc_after_training": "max",
            "validation_loss": "min",
            "validation_acc": "max",
        }

        for key, stats in global_stats.items():
            if key == "round_step":
                continue
            self.log_npy(key.lower().replace(" ", "_"), stats)
            if key in basic_stats:
                if basic_stats[key] == "min":
                    self.log_min_stats_per_client(stats, 1, key)
                else:
                    self.log_max_stats_per_client(stats, 1, key)
