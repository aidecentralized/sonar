# scheduler.py
"""
This module manages the orchestration of federated learning experiments.
"""

import os
import time
from typing import Any, Dict, List

import torch

import random
import numpy

from algos.base_class import BaseNode
from algos.fl import FedAvgClient, FedAvgServer
from algos.isolated import IsolatedServer
from algos.fl_assigned import FedAssClient, FedAssServer
from algos.fl_isolated import FedIsoClient, FedIsoServer
from algos.fl_weight import FedWeightClient, FedWeightServer
from algos.fl_static import FedStaticNode, FedStaticServer
from algos.swarm import SWARMClient, SWARMServer
from algos.DisPFL import DisPFLClient, DisPFLServer
from algos.def_kt import DefKTClient, DefKTServer
from algos.fedfomo import FedFomoClient, FedFomoServer
from algos.L2C import L2CClient, L2CServer
from algos.MetaL2C import MetaL2CClient, MetaL2CServer
from algos.fl_central import CentralizedCLient, CentralizedServer
from algos.fl_data_repr import FedDataRepClient, FedDataRepServer
from algos.fl_val import FedValClient, FedValServer

from utils.communication.comm_utils import CommunicationManager
from utils.config_utils import load_config, process_config
from utils.log_utils import copy_source_code, check_and_create_path

# Mapping of algorithm names to their corresponding client and server classes so that they can be consumed by the scheduler later on.
algo_map: Dict[str, List[FedAvgClient]] = { # type: ignore
    "fedavg": [FedAvgServer, FedAvgClient],
    "isolated": [IsolatedServer, IsolatedServer],
    "fedass": [FedAssServer, FedAssClient],
    "fediso": [FedIsoServer, FedIsoClient],
    "fedweight": [FedWeightServer, FedWeightClient],
    "fedstatic": [FedStaticServer, FedStaticNode],
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


def get_node(
    config: Dict[str, Any], rank: int, comm_utils: CommunicationManager
) -> BaseNode:
    algo_name = config["algo"]
    node_class = algo_map[algo_name][rank > 0]
    node = node_class(config, comm_utils) # type: ignore
    return node # type: ignore


class Scheduler:
    """Manages the overall orchestration of experiments"""

    def __init__(self) -> None:
        pass

    def install_config(self) -> None:
        self.config: Dict[str, Any] = process_config(self.config)

    def assign_config_by_path(
        self,
        sys_config_path: str,
        algo_config_path: str,
        is_super_node: bool | None = None,
        host: str | None = None,
    ) -> None:
        self.sys_config = load_config(sys_config_path)
        if is_super_node:
            self.sys_config["comm"]["rank"] = 0
        else:
            self.sys_config["comm"]["host"] = host
            self.sys_config["comm"]["rank"] = None
        self.config = {}
        self.config.update(self.sys_config)

    def merge_configs(self) -> None:
        self.config.update(self.sys_config)
        node_name = "node_{}".format(self.communication.get_rank())
        self.algo_config = self.sys_config["algos"][node_name]
        self.config.update(self.algo_config)
        self.config["dropout_dicts"] = self.sys_config.get("dropout_dicts", {}).get(node_name, {})

    def initialize(self, copy_souce_code: bool = True) -> None:
        assert self.config is not None, "Config should be set when initializing"
        self.communication = CommunicationManager(self.config)
        self.config["comm"]["rank"] = self.communication.get_rank()
        # Base clients modify the seed later on
        seed = self.config["seed"]
        torch.manual_seed(seed)  # type: ignore
        random.seed(seed)
        numpy.random.seed(seed)
        self.merge_configs()

        if self.communication.get_rank() == 0:
            if copy_souce_code:
                copy_source_code(self.config)
            else:
                path = self.config["results_path"]
                check_and_create_path(path)
                os.mkdir(self.config["saved_models"])
                os.mkdir(self.config["log_path"])
        else:
            # wait for 10 seconds for the super node to create the directories
            # the reason we do not wait indefinitely is because we need
            # ordinary nodes to make directories as well if they are running
            # from a different machine
            print("Waiting for 10 seconds for the super node to create directories")
            time.sleep(10)

        self.node = get_node(
            self.config,
            rank=self.communication.get_rank(),
            comm_utils=self.communication,
        )

    def run_job(self) -> None:
        self.node.run_protocol()
        self.communication.finalize()
