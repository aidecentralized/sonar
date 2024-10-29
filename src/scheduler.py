"""
This module manages the orchestration of federated learning experiments.
"""

import os
import time
from typing import Any, Dict, List, Union

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
from algos.swift import SwiftNode, SwiftServer
from algos.DisPFL import DisPFLClient, DisPFLServer
from algos.def_kt import DefKTClient, DefKTServer
from algos.fedfomo import FedFomoClient, FedFomoServer
from algos.L2C import L2CClient, L2CServer
from algos.MetaL2C import MetaL2CClient, MetaL2CServer
from algos.fl_central import CentralizedCLient, CentralizedServer
from algos.fl_data_repr import FedDataRepClient, FedDataRepServer
from algos.fl_val import FedValClient, FedValServer
from algos.fl_push import FedAvgPushClient, FedAvgPushServer

from utils.communication.comm_utils import CommunicationManager
from utils.config_utils import load_config, process_config
from utils.log_utils import copy_source_code, check_and_create_path
from configs.sys_config import SystemConfig   # Ensure SystemConfig is correctly imported
from configs.algo_config import AlgoConfig
# Mapping of algorithm names to their corresponding client and server classes
algo_map: Dict[str, List[Any]] = {
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
    "swift": [SwiftServer, SwiftNode],
    "fedavgpush": [FedAvgPushServer, FedAvgPushClient],
}



def get_node(
    config: Union[SystemConfig, Dict[str, Any]], 
    rank: int, 
    comm_utils: CommunicationManager
) -> BaseNode:
    """
    Get the appropriate node based on algorithm and rank.
    
    Args:
        config: Either SystemConfig instance or config dictionary
        rank: Node rank (0 for server, >0 for clients)
        comm_utils: Communication manager instance
    
    Returns:
        Initialized node instance
    """
    if isinstance(config, SystemConfig):
        algo = config.algo_configs[f"node_{rank}"].algo
    else:
        algo = config["algo"]
        
    node_class = algo_map[algo][rank > 0]
    return node_class(config, comm_utils)


class Scheduler:
    """Manages the overall orchestration of experiments"""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.sys_config: Union[Dict[str, Any], SystemConfig] = {}
        self.algo_config: Optional[Union[Dict[str, Any], AlgoConfig]] = None
        self.communication: Optional[CommunicationManager] = None
        self.node: Optional[BaseNode] = None

    def install_config(self) -> None:
        """Process the configuration with defaults."""
        self.config = process_config(self.config)

    def assign_config_by_path(
        self,
        sys_config_path: str,
        algo_config_path: str,
        is_super_node: bool | None = None,
        host: str | None = None,
    ) -> None:
        """Load and assign configuration from files."""
        self.sys_config = load_config(sys_config_path)

        if isinstance(self.sys_config, SystemConfig):
            if not hasattr(self.sys_config, 'comm'):
                self.sys_config.comm = {}
            self.sys_config.comm["type"] = "GRPC"
            if is_super_node:
                self.sys_config.comm["rank"] = 0
            else:
                self.sys_config.comm["host"] = host
                self.sys_config.comm["rank"] = None
            self.config = vars(self.sys_config)
        else:
            if "comm" not in self.sys_config:
                self.sys_config["comm"] = {}
            if is_super_node:
                self.sys_config["comm"]["rank"] = 0
            else:
                self.sys_config["comm"]["host"] = host
                self.sys_config["comm"]["rank"] = None
            self.config = {}
            self.config.update(self.sys_config)

    def merge_configs(self) -> None:
        if isinstance(self.sys_config, SystemConfig):
            self.config.update(vars(self.sys_config))
            node_name = f"node_{self.communication.get_rank()}"
            self.algo_config = self.sys_config.algo_configs[node_name]
            if isinstance(self.algo_config, AlgoConfig):
                self.config.update(vars(self.algo_config))
            else:
                self.config.update(self.algo_config)
            dropout_dicts = getattr(self.sys_config, "dropout_dicts", {})
        else:
            self.config.update(self.sys_config)
            node_name = f"node_{self.communication.get_rank()}"
            self.algo_config = self.sys_config["algos"][node_name]
            self.config.update(self.algo_config)
            dropout_dicts = self.sys_config.get("dropout_dicts", {})

        self.config["dropout_dicts"] = dropout_dicts.get(node_name, {})
        self.config["load_existing"] = self.config.get("load_existing", False)  # Add default value here


    def initialize(self, should_copy_source_code: bool = True) -> None:
        """Initialize the scheduler and prepare for job execution."""
        assert self.config is not None, "Config should be set when initializing"
        
        self.communication = CommunicationManager(self.config)
        self.config["comm"]["rank"] = self.communication.get_rank()
        
        seed = self.config["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        
        self.merge_configs()

        if self.communication.get_rank() == 0:
            if should_copy_source_code:
                copy_source_code(self.config)  
            else:
                path = self.config["results_path"]
                check_and_create_path(path)
                os.makedirs(self.config["saved_models"], exist_ok=True)
                os.makedirs(self.config["log_path"], exist_ok=True)
        else:
            print("Waiting for 10 seconds for the super node to create directories")
            time.sleep(10)

        self.node = get_node(
            self.config,
            rank=self.communication.get_rank(),
            comm_utils=self.communication,
        )


    def run_job(self) -> None:
        """Execute the federated learning job."""
        if not hasattr(self, 'node') or self.node is None:
            raise ValueError("Node not initialized")
            
        self.node.run_protocol()
        if self.communication:
            self.communication.finalize()