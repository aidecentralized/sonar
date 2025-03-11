from algos.topologies.base import BaseTopology
from utils.types import ConfigType

from math import ceil
import networkx as nx


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

    def generate_graph(self, full:bool=False) -> None:
        """ if full is True, generate a fully connected graph """
        if full:
            while True:
                self.graph = nx.erdos_renyi_graph(self.num_users, self.p, seed=self.seed)
                if nx.is_connected(self.graph):
                    break
        else:
            self.graph = nx.erdos_renyi_graph(self.num_users, self.p, seed=self.seed)

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

class LineTopology(BaseTopology):
    """ test topology for debugging gradient disambiguation attack """
    def __init__(self, config: ConfigType, rank: int):
        super().__init__(config, rank)

    def generate_graph(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_node(0)
        for i in range(1, self.num_users):
            self.graph.add_node(i)
            self.graph.add_edge(i-1, i)

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
    if topology_name == "line":
        return LineTopology(config, rank)
    raise ValueError(f"Topology {topology_name} not implemented")