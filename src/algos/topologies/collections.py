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

    def generate_graph(self) -> None:
        self.graph = nx.circular_ladder_graph(self.num_users) # type: ignore

class TreeTopology(BaseTopology):
    def __init__(self, config: ConfigType, rank: int, children: int = 2):
        super().__init__(config, rank)
        self.children = children

    def generate_graph(self) -> None:
        self.graph = nx.Graph()

        self.graph.add_node(0)  # Start with a root node
        nodes = [0]
        
        node_count = 1
        while node_count < self.num_users:
            new_nodes = []
            for node in nodes:
                # Add children to the current node
                for _ in range(self.children):
                    if node_count >= self.num_users:
                        break  # Stop if we've reached the total number of nodes
                    self.graph.add_node(node_count)
                    self.graph.add_edge(node, node_count)
                    new_nodes.append(node_count)
                    node_count += 1
            nodes = new_nodes  # Move on to the next level of children



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
    raise ValueError(f"Topology {topology_name} not implemented")