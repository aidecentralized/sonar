# scheduler.py
"""
This module manages the orchestration of federated learning experiments.
"""

import os
import random

from mpi4py import MPI
import torch

import numpy as np

import random
import numpy

from algos.base_class import BaseNode
from algos.fl import FedAvgClient, FedAvgServer
from algos.isolated import IsolatedServer
from algos.fl_assigned import FedAssClient, FedAssServer
from algos.fl_isolated import FedIsoClient, FedIsoServer
from algos.fl_weight import FedWeightClient, FedWeightServer
from algos.fl_static import FedStaticClient, FedStaticServer
from algos.swarm import SWARMClient, SWARMServer
from algos.DisPFL import DisPFLClient, DisPFLServer
from algos.def_kt import DefKTClient, DefKTServer
from algos.fedfomo import FedFomoClient, FedFomoServer
from algos.L2C import L2CClient, L2CServer
from algos.MetaL2C import MetaL2CClient, MetaL2CServer
from algos.fl_central import CentralizedCLient, CentralizedServer
from algos.fl_data_repr import FedDataRepClient, FedDataRepServer
from algos.fl_val import FedValClient, FedValServer

from utils.log_utils import check_and_create_path
from utils.config_utils import load_config, process_config

from utils.log_utils import copy_source_code, check_and_create_path
from utils.config_utils import load_config, process_config, get_device_ids
import os


# Mapping of algorithm names to their corresponding client and server classes so that they can be consumed by the scheduler later on.
algo_map = {
    "fedavg": [FedAvgServer, FedAvgClient],
    "isolated": [IsolatedServer],

#    "fedran": [FedRanServer, FedRanClient],
#    "fedgrid": [FedGridServer, FedGridClient],
#    "fedtorus": [FedTorusServer, FedTorusClient],
    "fedass": [FedAssServer, FedAssClient],
    "fediso": [FedIsoServer, FedIsoClient],
    "fedweight": [FedWeightServer, FedWeightClient],
#    "fedring": [FedRingServer, FedRingClient],
    "fedstatic": [FedStaticServer, FedStaticClient],

    "swarm": [SWARMServer, SWARMClient],
    "dispfl": [DisPFLServer, DisPFLClient],
    "defkt": [DefKTServer, DefKTClient],
    "fedfomo": [FedFomoServer, FedFomoClient],
    "l2c": [L2CServer, L2CClient],
    "metal2c": [MetaL2CServer, MetaL2CClient],
    "centralized": [CentralizedServer, CentralizedCLient],
    "feddatarepr": [FedDataRepServer, FedDataRepClient],
    "fedval": [FedValServer, FedValClient],
}

def get_node(config: dict, rank) -> BaseNode:
    algo_name = config["algo"]
    return algo_map[algo_name][rank > 0](config)
  
class Scheduler():
    """ Manages the overall orchestration of experiments
    """

    def __init__(self) -> None:
        pass

    def install_config(self) -> None:
        self.config = process_config(self.config)

    def assign_config_by_path(self, sys_config_path, algo_config_path):
        self.sys_config = load_config(sys_config_path)
        self.algo_config = load_config(algo_config_path)
        self.merge_configs()

    def merge_configs(self):
        self.config = {}
        self.config.update(self.sys_config)
        self.config.update(self.algo_config)

    def initialize(self, copy_souce_code=True) -> None:
        assert self.config is not None, "Config should be set when initializing"

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Base clients modify the seed later on
        seed = self.config["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)

        if rank == 0:
            if copy_souce_code:
                copy_source_code(self.config)
            else:
                path = self.config["results_path"]
                check_and_create_path(path)
                os.mkdir(self.config["saved_models"])
                os.mkdir(self.config["log_path"])

        self.node = get_node(self.config, rank=rank)

    def run_job(self) -> None:
        self.node.run_protocol()
