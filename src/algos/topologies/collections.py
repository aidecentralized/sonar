from typing import List
from algos.topologies.base import BaseTopology
from utils.types import ConfigType

from math import ceil, log2
import networkx as nx
from algos.topologies.base_exponential import OnePeerExponentialGraph, HyperHyperCube, SimpleBaseGraph, BaseGraph


class RingTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self) -> None:
        self.graph = nx.cycle_graph(self.num_users) # type: ignore


class StarTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self) -> None:
        self.graph = nx.star_graph(self.num_users - 1) # type: ignore


class FullyConnectedTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self) -> None:
        self.graph = nx.complete_graph(self.num_users) # type: ignore


class GridTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        if self.num_users**0.5 != int(self.num_users**0.5):
            raise ValueError("Number of users should be a perfect square for grid topology")

    def generate_graph(self) -> None:
        self.graph = nx.grid_2d_graph(ceil(self.num_users**0.5), ceil(self.num_users**0.5)) # type: ignore


class TorusTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        if self.num_users**0.5 != int(self.num_users**0.5):
            raise ValueError("Number of users should be a perfect square for grid topology")

    def generate_graph(self) -> None:
        self.graph = nx.grid_2d_graph(ceil(self.num_users**0.5), ceil(self.num_users**0.5), periodic=True) # type: ignore

class CircleLadderTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        if self.num_users%2 != 0:
            raise ValueError("Number of users should be an even number for circle ladder topology")

    def generate_graph(self) -> None:
        self.graph = nx.circular_ladder_graph(int(self.num_users/2)) # type: ignore

class TreeTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int, children: int = 2):
        super().__init__(config, rank)
        self.children = children

    def generate_graph(self) -> None:
        self.graph = nx.full_rary_tree(self.children, self.num_users) # type: ignore

class LadderTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        if self.num_users%2 != 0:
            raise ValueError("Number of users should be an even number for ladder topology")

    def generate_graph(self) -> None:
        self.graph = nx.ladder_graph(int(self.num_users/2)) # type: ignore

class WheelTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self) -> None:
        self.graph = nx.wheel_graph(self.num_users) # type: ignore

class BipartiteTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        if self.num_users<2:
            raise ValueError("Need at least 2 users for a bipartite topology")

    def generate_graph(self) -> None:
        self.graph = nx.turan_graph(self.num_users,2) # type: ignore

class BarbellTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        b: int = config["topology"]["b"] # type: ignore
        super().__init__(config, rank)
        self.barbell_size = b
        self.path_length = self.num_users - (2*self.barbell_size)
        if self.barbell_size<=2:
            raise ValueError("Need at least 2 nodes b in each barbell")
        elif self.path_length<0:
            raise ValueError("Need at least path length 0 between barbells")
    

    def generate_graph(self) -> None:
        self.graph = nx.barbell_graph(self.barbell_size, self.path_length) # type: ignore



######### Random Graphs #########
class ErdosRenyiTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        p: float = config["topology"]["p"] # type: ignore
        super().__init__(config, rank)
        self.p = p
        self.seed = config["seed"]

    def generate_graph(self) -> None:
        self.graph = nx.erdos_renyi_graph(self.num_users, self.p, self.seed)


class WattsStrogatzTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        k: int = config["topology"]["k"] # type: ignore
        p: float = config["topology"]["p"] # type: ignore
        super().__init__(config, rank)
        self.k = k
        self.p = p
        self.seed = config["seed"]

    def generate_graph(self) -> None:
        self.graph = nx.watts_strogatz_graph(self.num_users, self.k, self.p, self.seed) # type: ignore

class RandomRegularTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        d: int = config["topology"]["d"] # type: ignore
        super().__init__(config, rank)
        self.d = d
        self.seed = config["seed"]
        if (d*self.num_users)%2 != 0:
            raise ValueError("d * number of users must be even to make a valid graph")

    def generate_graph(self) -> None:
        self.graph = nx.random_regular_graph(self.d, self.num_users, self.seed) # type: ignore


class DynamicGraph(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.itr = -1

    def _convert_labels_to_int(self) -> None:
        """
        Performs two operations:
        1. Convert the labels of the graph to integers - useful for grid like graphs where labels are tuples
        2. Convert the graph to use 1-based indexing - useful for indexing because we reserve 0 for the super node
        """
        if self.graph is None:
            raise ValueError("Graph not initialized")
        self.graph = [nx.convert_node_labels_to_integers(graph, first_label=1) for graph in self.graph]  # type: ignore

    def generate_graph(self) -> None:
        raise NotImplementedError

    def _convert_weight_matrices_to_graph(self, w_list):
        g_list = []
        for w in w_list:
            G = nx.from_numpy_array(w.numpy(), create_using=nx.DiGraph)
            G.remove_edges_from(nx.selfloop_edges(G))
            g_list.append(G)
        return g_list
    
    def get_in_neighbors(self):
        """
        Returns the list of in neighbours of the current node
        """
        self.itr += 1
        return list(self.graph[self.itr%len(self.graph)].predecessors(self.rank))

    def get_out_neighbors(self, i):
        """
        Returns the list of out neighbours of the current node
        """
        self.itr += 1
        return list(self.graph[self.itr%len(self.graph)].successors(i))
    
    def get_all_neighbours(self) -> List[int]:
        self.itr += 1
        return self.get_in_neighbors(self.rank)

class DynamicBaseGraph(DynamicGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self, class_name, *args, **kwargs) -> None:
        w_list = class_name(*args, **kwargs).w_list
        self.graph = self._convert_weight_matrices_to_graph(w_list)


class OnePeerExponentialTopology(DynamicBaseGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self) -> None:
        super().generate_graph(OnePeerExponentialGraph, self.num_users)

    # def generate_graph(self) -> None:
    #     self.graph = [nx.DiGraph() for _ in range(self.num_users)]

    #     num_neighbors = int(log2(self.num_users-1))

    #     for j in range(num_neighbors+1):
    #         for i in range(self.num_users):
    #             self.graph[j].add_edge(self.rank, (i+2**j)%self.num_users)

class HyperHyperCubeTopology(DynamicBaseGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.seed = config["seed"]
        self.max_degree = config["topology"].get("max_degree", 1)

    def generate_graph(self) -> None:
        super().generate_graph(HyperHyperCube, self.num_users, self.max_degree, self.seed)

class SimpleBaseGraphTopology(DynamicBaseGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.seed = config["seed"]
        self.max_degree = config["topology"].get("max_degree", 1)
        self.inner_edges = config["topology"].get("inner_edges", True)

    def generate_graph(self) -> None:
        super().generate_graph(SimpleBaseGraph, self.num_users, self.max_degree, self.seed, self.inner_edges)

class BaseGraphTopology(DynamicBaseGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.seed = config["seed"]
        self.max_degree = config["topology"].get("max_degree", 1)
        self.inner_edges = config["topology"].get("inner_edges", True)

    def generate_graph(self) -> None:
        super().generate_graph(BaseGraph, self.num_users, self.max_degree, self.seed, self.inner_edges)

def select_topology(config: ConfigType, rank: int) -> BaseTopology:
    """
    Selects the topology based on the configuration.
    """
    topology_name = config["topology"]["name"] # type: ignore
    if topology_name == "ring":
        return RingTopology(config, rank)
    if topology_name == "star":
        return StarTopology(config, rank)
    if topology_name == "grid":
        return GridTopology(config, rank)
    if topology_name == "torus":
        return TorusTopology(config, rank)
    if topology_name == "fully_connected":
        return FullyConnectedTopology(config, rank)
    if topology_name == "circle_ladder":
        return CircleLadderTopology(config, rank)
    if topology_name == "erdos_renyi":
        return ErdosRenyiTopology(config, rank)
    if topology_name == "watts_strogatz":
        return WattsStrogatzTopology(config, rank)
    if topology_name == "tree":
        return TreeTopology(config, rank)
    if topology_name == "ladder":
        return LadderTopology(config, rank)
    if topology_name == "wheel":
        return WheelTopology(config, rank)
    if topology_name == "bipartite":
        return BipartiteTopology(config, rank)
    if topology_name == "random_regular":
        return RandomRegularTopology(config, rank)
    if topology_name == "barbell":
        return BarbellTopology(config, rank)
    if topology_name == "one_peer_exponential":
        return OnePeerExponentialTopology(config, rank)
    if topology_name == "hyper_hypercube":
        return HyperHyperCubeTopology(config, rank)
    if topology_name == "simple_base_graph":
        return SimpleBaseGraphTopology(config, rank)
    if topology_name == "base_graph":
        return BaseGraphTopology(config, rank)
    raise ValueError(f"Topology {topology_name} not implemented")