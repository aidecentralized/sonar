import os
import pickle
import shutil
import logging
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from shutil import copytree, copy2
from glob import glob
from PIL import Image
import numpy as np

# Normalize an image


def deprocess(img):
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img = inv_normalize(img)
    img = 255 * img
    return img.type(torch.uint8)


def check_and_create_path(path):
    if os.path.isdir(path):
        print("Experiment in {} already present".format(path))
        done = False
        while not done:
            inp = input("Press e to exit, r to replace it: ")
            if inp == "e":
                exit()
            elif inp == "r":
                done = True
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                print("Input not understood")
                # exit()
    else:
        os.makedirs(path)


def copy_source_code(config: dict) -> None:
    """Copy source code to experiment folder
    This happens only once at the start of the experiment
    This is to ensure that the source code is snapshoted at the start of the experiment
    for reproducibility purposes
    Args:
        config (dict): [description]
    """
    path = config["results_path"]
    print("exp path:", path)
    if config["load_existing"]:
        print("Continue with loading checkpoint")
        return
    else:
        # throw a prompt
        check_and_create_path(path)
        # the last folder is the path where all the expts are stored
        denylist = ["./__pycache__/",
                    "./.ipynb_checkpoints/",
                    "./expt_dump/",
                    "./helper_scripts/",
                    "./imgs/",
                    "./expt_dump_old/",
                    "./comparison_plots/",
                    "./toy_exp/",
                    "./toy_exp_ml/",
                    "./toy_exp.py",
                    "./toy_exp_ml.py"
                    '/'.join(path.split('/')[:-1]) + '/']
        folders = glob(r'./*/')
        print(denylist, folders)

        # For copying python files
        for file_ in glob(r'./*.py'):
            copy2(file_, path)

        # For copying json files
        for file_ in glob(r'./*.json'):
            copy2(file_, path)

        for folder in folders:
            if folder not in denylist:
                # Remove first char which is . due to the glob
                copytree(folder, path + folder[1:])

        # For saving models in the future
        os.mkdir(config['saved_models'])
        os.mkdir(config['log_path'])
        print("source code copied to exp_dump")


class LogUtils():
    def __init__(self, config) -> None:
        log_dir, load_existing = config["log_path"], config["load_existing"]
        log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
                     "%(filename)s::%(lineno)d::%(message)s"
        logging.basicConfig(filename="{log_path}/log_console.log".format(
            log_path=log_dir),
            level='DEBUG', format=log_format)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.log_dir = log_dir
        self.init_tb(load_existing)
        self.init_npy()
        self.init_summary()

    def init_summary(self):
        # Open a txt file to write summary
        self.summary_file = open(f"{self.log_dir}/summary.txt", "w")

    def init_tb(self, load_existing):
        tb_path = self.log_dir + "/tensorboard"
        # if not os.path.exists(tb_path) or not os.path.isdir(tb_path):
        if not load_existing:
            os.makedirs(tb_path)
        self.writer = SummaryWriter(tb_path)

    def init_npy(self):
        npy_path = self.log_dir + "/npy"
        if not os.path.exists(npy_path) or not os.path.isdir(npy_path):
            os.makedirs(npy_path)

    def log_image(self, imgs: torch.Tensor, key, iteration):
        # imgs = deprocess(imgs.detach().cpu())[:64]
        grid_img = make_grid(
            imgs.detach().cpu(),
            normalize=True,
            scale_each=True)
        # Save the grid image using torchvision api
        save_image(grid_img, f"{self.log_dir}/{iteration}_{key}.png")
        # Save the grid image using tensorboard api
        self.writer.add_image(key, grid_img.numpy(), iteration)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)

    def log_npy(self, key, value):
        np.save(f"{self.log_dir}/npy/{key}.npy", value)

    def log_max_stats_per_client(self, stats_per_client, round_step, metric):
        self.__log_stats_per_client__(
            stats_per_client, round_step, metric, is_max=True)

    def log_min_stats_per_client(self, stats_per_client, round_step, metric):
        self.__log_stats_per_client__(
            stats_per_client, round_step, metric, is_max=False)

    def __log_stats_per_client__(
            self,
            stats_per_client,
            round_step,
            metric,
            is_max=False):
        if is_max:
            best_round_per_client = np.argmax(
                stats_per_client, axis=1) * round_step
            best_val_per_client = np.max(stats_per_client, axis=1)
        else:
            best_round_per_client = np.argmin(
                stats_per_client, axis=1) * round_step
            best_val_per_client = np.min(stats_per_client, axis=1)

        minmax = 'max' if is_max else 'min'
        # Write to summary file
        self.summary_file.write(
            f"============== {minmax} {metric} per client ==============\n")
        for client_idx, (best_round, best_val) in enumerate(
                zip(best_round_per_client, best_val_per_client)):
            self.summary_file.write(
                f"Client {client_idx+1} : {best_val} at round {best_round}\n")
        self.summary_file.write(
            f"Mean of {minmax} {metric} : {np.mean(best_val_per_client)}, quantiles: {np.quantile(best_val_per_client, [0.25, 0.75])}\n")

    def log_tb_round_stats(self, round_stats, stats_to_exclude, round):
        stats_key = round_stats[0].keys()
        for key in stats_key:
            if key not in stats_to_exclude:
                average = 0
                for client_id, stats in enumerate(round_stats, 1):
                    self.log_tb(f"{key}/client{client_id}", stats[key], round)
                    average += stats[key]
                average /= len(round_stats)
                self.log_tb(f"{key}/clients", average, round)

    def log_experiments_stats(self, gloabl_stats):

        basic_stats = {
            "train_loss": "min",
            "train_acc": "max",
            "test_acc": "max",
            "test_acc_before_training": "max",
            "test_acc_after_training": "max",
            "validation_loss": "min",
            "validation_acc": "max",
        }

        for key, stats in gloabl_stats.items():
            if key == "round_step":
                continue
            self.log_npy(key.lower().replace(" ", "_"), stats)
            if key in basic_stats:
                if basic_stats[key] == "min":
                    self.log_min_stats_per_client(stats, 1, key)
                else:
                    self.log_max_stats_per_client(stats, 1, key)
