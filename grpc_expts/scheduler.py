import torch, random, numpy
from algos.base_class import BaseNode
from utils.log_utils import copy_source_code, check_and_create_path
from utils.config_utils import load_config, process_config
from grpc_expts.utils.comm_utils import CommUtils
from algos.base_class import BaseNode
from algos.fl import FedAvgClient, FedAvgServer
from algos.isolated import IsolatedServer
from algos.fl_random import FedRanClient, FedRanServer
from algos.fl_assigned import FedAssClient, FedAssServer
from algos.fl_isolated import FedIsoClient, FedIsoServer
from algos.fl_weight import FedWeightClient, FedWeightServer
from algos.swarm import SWARMClient, SWARMServer
from algos.DisPFL import DisPFLClient, DisPFLServer
from algos.def_kt import DefKTClient,DefKTServer
from algos.fedfomo import FedFomoClient, FedFomoServer
from algos.L2C import L2CClient, L2CServer
from algos.MetaL2C import MetaL2CClient, MetaL2CServer
from algos.fl_central import CentralizedCLient, CentralizedServer
from algos.fl_data_repr import FedDataRepClient, FedDataRepServer
from algos.fl_val import FedValClient, FedValServer
from utils.log_utils import copy_source_code, check_and_create_path
from utils.config_utils import load_config, process_config
import os

algo_map = {
    "fedavg": [FedAvgServer, FedAvgClient],
    "isolated": [IsolatedServer],
    "fedran": [FedRanServer,FedRanClient],
    "fedass": [FedAssServer, FedAssClient],
    "fediso": [FedIsoServer,FedIsoClient],
    "fedweight": [FedWeightServer,FedWeightClient],
    "swarm" : [SWARMServer, SWARMClient],
    "dispfl": [DisPFLServer, DisPFLClient],
    "defkt": [DefKTServer,DefKTClient],
    "fedfomo": [FedFomoServer, FedFomoClient],
    "l2c": [L2CServer,L2CClient],
    "metal2c": [MetaL2CServer,MetaL2CClient],
    "centralized": [CentralizedServer, CentralizedCLient],
    "feddatarepr": [FedDataRepServer, FedDataRepClient],
    "fedval": [FedValServer, FedValClient],
}

def get_node(config, client_id):
    algo_name = config["algo"]
    return algo_map[algo_name][client_id > 0](config)

class Scheduler:
    def __init__(self):
        self.comm_utils = None
        self.node = None

    def assign_config_by_path(self, config_path):
        self.config = load_config(config_path)

    def install_config(self):
        self.config = process_config(self.config)

    def initialize(self, copy_source_code=True):
        assert self.config is not None, "Config should be set when initializing"
        seed = self.config["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        self.comm_utils = CommUtils(server_address=self.config["server_address"])
        self.node = get_node(self.config, client_id=self.comm_utils.client_id)
        if self.comm_utils.client_id == 0 and copy_source_code:
            copy_source_code(self.config)
            check_and_create_path(self.config["results_path"])

    def run_job(self):
        self.node.run_protocol()
