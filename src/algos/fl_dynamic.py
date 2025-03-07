"""
Module for FedStaticClient and FedStaticServer in Federated Learning.
"""
from typing import Any, Dict, List
from utils.communication.comm_utils import CommunicationManager
import torch

from algos.base_class import BaseFedAvgClient
from algos.topologies.collections import FullyConnectedTopology
from utils.types import TorchModelType

# TODO: Create a separate similarity evaluation class that supports different similarity metrics
# 1.b Similar based on KL divergence between the output distribution

# TODO: Create a separate sampling class that supports different sampling strategies
# 2. Sample based on a probability distribution
# 3. Sample based epsilon-greedy strategy

# TODO: Clustering based samplers
# Still requires a similarity metric
# 1. K-means clustering
# 2. Affinity Propagation

def select_smallest_k(collab_weights: List[float], k: int) -> List[int]:
    """
    Selects the top k smallest values from the list
    """
    return sorted(range(len(collab_weights)), key=lambda i: collab_weights[i])[:k]

class DynamicTopology(FullyConnectedTopology):
    """
    The FullyConnectedTopology class is one of the
    static topologies that assumes all nodes are connected
    which is needed for the Dynamic topology because in most
    cases, it needs to query all nodes to choose the next set of neighbors.
    """
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager, base_node: BaseFedAvgClient) -> None:
        super().__init__(config, comm_utils.get_rank())
        self.comm_utils = comm_utils
        self.similarity = config.get("topology").get("comparison") # type: ignore
        self.sampling = config.get("topology").get("sampling") # type: ignore
        self.base_node = base_node

    def l2_distance(self, other_wts: TorchModelType) -> float:
        own: TorchModelType = self.base_node.get_model_weights()["model"] # type: ignore
        distance = 0.
        for key in own.keys():
            if key.endswith("num_batches_tracked"):
                continue
            distance += torch.norm(own[key] - other_wts[key]).item() # type: ignore
        return distance # type: ignore

    def similarity_loss(self, own_outputs: torch.Tensor, other_wts: TorchModelType) -> float:
        """
        Returns the similarity of the neighbors
        based on the similarity metric.
        """
        dloader = self.base_node.dloader # type: ignore
        device = self.base_node.device

        own_model: TorchModelType = self.base_node.get_model_weights()["model"] # type: ignore
        self.base_node.set_model_weights(other_wts)
        outputs = self.base_node.model_utils.forward_pass(self.base_node.model, dloader, device)
        self.base_node.set_model_weights(own_model)
    
        # compute l2 distance between the outputs
        return torch.norm(outputs - own_outputs).item() # type: ignore

    def get_neighbor_model_wts(self) -> List[Dict[str, TorchModelType]]:
        """
        Returns the model weights of the neighbors
        that will be used by the neighbor sampler.
        In its current version, it will get weights
        from all the neighbors because that's how
        most dynamic topologies work.
        """
        neighbor_models: List[Dict[str, TorchModelType]] = self.comm_utils.all_gather(ignore_super_node=True)
        return neighbor_models

    def get_neighbor_similarity(self, others_wts: List[Dict[str, TorchModelType]]) -> List[float]:
        """
        Returns the similarity of the neighbors
        based on the similarity metric.
        """
        with torch.no_grad():
            similarity_wts: List[float] = []
            for item in others_wts:
                model_wts = item["model"]
                if self.similarity == "weights_l2":
                    similarity_wts.append(self.l2_distance(model_wts))
                elif self.similarity == "loss":
                    own_outputs = self.base_node.model_utils.forward_pass(self.base_node.model, self.base_node.dloader, self.base_node.device)
                    similarity_wts.append(self.similarity_loss(own_outputs, model_wts))
                else:
                    raise ValueError("Similarity metric {} not implemented".format(self.similarity))
        return similarity_wts

    def sample_neighbours(self, k: int, mode: str|None = None) -> List[int]:
        """
        We perform neighbor sampling after
        we have the similarity weights of all the neighbors.
        """
        assert mode is None or mode == "pull", "Only pull mode is supported for dynamic topology"
        if self.sampling == "closest":
            return select_smallest_k(self.similarity_wts, k)
        else:
            raise ValueError("Sampling strategy {} not implemented".format(self.sampling))

    def get_selected_collabs(self, neighbors: List[int]) -> List[int]:
        """
        Returns the selected collaborators
        The implementation of this stands at the edge of a knife.
        Basically, to get the rank of collaborators, we need to
        first increment the list of neighbors by one index because
        zero belongs to the super node. Second, we need to
        add one to the rank of the nodes that have rank bigger
        than the current node.
        FIXME: Change how we implement receive in comm_utils
        to fix this issue. All this wouldn't be needed if we
        obtain rank and weights in the receive function. For now
        we rely on the list index as a proxy for the rank.
        TODO: We already have sender key in the weights and hence we can
        use that to get the rank of the node.
        """
        return [neighbor + 2 if neighbor >= self.rank - 1 else neighbor + 1 for neighbor in neighbors]
        
    def recv_and_agg(self, k: int) -> List[int]:
        """
        all_wts = self.get_neighbor_model_wts()
        similarity_wts = self.get_neighbor_similarity(all_wts)
        collab_wts = get_collab_wts(collab_weights, all_wts)
        return collab_wts
        """
        # first we insert 
        all_wts = self.get_neighbor_model_wts()
        similarity_wts = self.get_neighbor_similarity(all_wts)
        self.similarity_wts = similarity_wts
        neighbors = self.sample_neighbours(k)
        selected_wts = [all_wts[i] for i in neighbors]
        self.base_node.aggregate(selected_wts) # type: ignore
        # apply the aggregation of the weights
        selected_collabs = self.get_selected_collabs(neighbors)
        return selected_collabs


class FedDynamicNode(BaseFedAvgClient):
    """
    Federated Static Client Class.
    """

    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        super().__init__(config, comm_utils)
        self.topology = DynamicTopology(config, comm_utils, self)
        self.topology.initialize()

    def get_representation(self, **kwargs: Any) -> Dict[str, int|Dict[str, Any]]:
        """
        Returns the model weights as representation.
        """
        return self.get_model_weights()

    def run_protocol(self) -> None:
        """
        Runs the federated learning protocol for the client.
        """
        stats: Dict[str, Any] = {}
        print(f"Client {self.node_id} ready to start training")

        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config.get("epochs_per_round", 1)

        for it in range(start_round, total_rounds):
            self.round_init()

            # Train locally and send the representation to the server
            stats["train_loss"], stats["train_acc"], stats["train_time"] = self.local_train(
                    it, epochs_per_round
                )            
            self.local_round_done()

            # Collect the representations from all other nodes from the server
            collabs = self.topology.recv_and_agg(self.num_collaborators)

            self.stats["neighbors"] = collabs
            self.local_test()
            self.round_finalize()



class FedDynamicServer(BaseFedAvgClient):
    """
    Federated Static Server Class. It does not do anything.
    It just exists to keep the code compatible across different algorithms.
    """
    def __init__(
        self, config: Dict[str, Any], comm_utils: CommunicationManager
    ) -> None:
        pass

    def run_protocol(self) -> None:
        pass
