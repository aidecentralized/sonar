from typing import List
from algos.topologies.base import BaseTopology
from utils.types import ConfigType

from math import ceil, log2
import networkx as nx
from base_exponential import OnePeerExponentialGraph, HyperHyperCube, SimpleBaseGraph, BaseGraph


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

    def generate_graph(self) -> None:
        self.graph = nx.circular_ladder_graph(self.num_users) # type: ignore

class TreeTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int, children: int = 2):
        super().__init__(config, rank)
        self.children = children

    def generate_graph(self) -> None:
        self.graph = nx.full_rary_tree(self.children, self.num_users)



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

class DynamicGraph(BaseTopology):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.itr = -1

    def generate_graph(self) -> None:
        raise NotImplementedError

    def _convert_weight_matrices_to_graph(self, w_list):
        return [nx.from_numpy_array(w, create_using=nx.DiGraph) for w in w_list]        
    
    def get_in_neighbors(self):
        """
        Returns the list of in neighbours of the current node
        """
        self.itr += 1
        return self.graph[self.itr%len(self.graph)].predecessors(self.rank)

    def get_out_neighbors(self, i):
        """
        Returns the list of out neighbours of the current node
        """
        self.itr += 1
        return self.graph[self.itr%len(self.graph)].successors(i)
    
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
        self.max_degree = config["topology"]["max_degree"]

    def generate_graph(self) -> None:
        super().generate_graph(HyperHyperCube, self.num_users, self.max_degree, self.seed)

class SimpleBaseGraphTopology(DynamicBaseGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.seed = config["seed"]
        self.max_degree = config["topology"]["max_degree"]
        self.inner_edges = config["topology"].get("inner_edges", True)

    def generate_graph(self) -> None:
        super().generate_graph(SimpleBaseGraph, self.num_users, self.max_degree, self.seed, self.inner_edges)

class BaseGraphTopology(DynamicBaseGraph):
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)
        self.seed = config["seed"]
        self.max_degree = config["topology"]["max_degree"]
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
    if topology_name == "one_peer_exponential":
        return OnePeerExponentialTopology(config, rank)
    if topology_name == "hyper_hypercube":
        return HyperHyperCubeTopology(config, rank)
    if topology_name == "simple_base_graph":
        return SimpleBaseGraphTopology(config, rank)
    if topology_name == "base_graph":
        return BaseGraphTopology(config, rank)
    raise ValueError(f"Topology {topology_name} not implemented")