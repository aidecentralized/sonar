"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, OrderedDict, List
from collections import OrderedDict, defaultdict

from utils.communication.comm_utils import CommunicationManager

from algos.base_class import BaseFedAvgClient
from algos.topologies.collections import select_topology
from utils.data_utils import get_dataset

class FedStaticNode(BaseFedAvgClient):
    """
    Federated Static Client Class.
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.topology = select_topology(config, self.node_id)
        self.topology.initialize()


    def get_representation(self, **kwargs: Any) -> Dict[str, int|Dict[str, Any]]:
        """
        Returns a list of neighbours for the client.
        """
        return self.get_model_weights()


    def get_neighbors(self) -> List[int]:
        """
        Returns a list of neighbours for the clients
        """
        neighbors = self.topology.sample_neighbours(self.num_collaborators, mode="pull")
        self.stats["neighbors"] = neighbors  # type: ignore, where the hell self.stats is coming from

        return neighbors

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        print(f"Client {self.node_id} ready to start training")
        start_round = self.config.get("start_round", 0)
        if start_round != 0:
            raise NotImplementedError(
                "Start round different from 0 not implemented yet"
            )
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)

        for it in range(start_round, total_rounds):
            self.round_init()

            # Train locally and send the representation to the server
            self.local_train(
                    it, epochs_per_round
                )            
            self.local_round_done()
            # Collect the representations from all other nodes from the server

            neighbors = self.get_neighbors()
            # TODO: Log the neighbors
            self.receive_and_aggregate(neighbors)
            # evaluate the model on the test data
            # Inside FedStaticNode.run_protocol()
            self.local_test()

            self.round_finalize()

class FedStaticServer(BaseFedAvgClient):
    """
    Federated Static Server Class. It does not do anything.
    It just exists to keep the code compatible across different algorithms.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        self.comm_utils = comm_utils
        self.node_id = self.comm_utils.get_rank()
        self.comm_utils.register_node(self)
        self.is_working = True
        if isinstance(config["dset"], dict):
            if self.node_id != 0:
                config["dset"].pop("0") # type: ignore
            self.dset = str(config["dset"][str(self.node_id)]) # type: ignore
            config["dpath"] = config["dpath"][self.dset]
        else:
            self.dset = config["dset"]
        print(f"Node {self.node_id} getting dset at {self.dset}")
        self.dset_obj = get_dataset(self.dset, dpath=config["dpath"])

    def run_protocol(self) -> None:
        pass
