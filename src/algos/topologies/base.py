from abc import ABC, abstractmethod
from typing import List

import numpy as np
import networkx as nx

from utils.types import ConfigType

class BaseTopology(ABC):
    """
    Base class for network topologies
    """

    def __init__(self, config: ConfigType, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.num_users: int = self.config["num_users"] # type: ignore
        self.graph: nx.Graph | None = None

    @abstractmethod
    def generate_graph(self) -> None:
        """
        Generate the graph using the networkX library
        and store it in the self.graph attribute
        NetworkX has a lot of built-in functions to generate graphs
        Use this url - https://networkx.org/documentation/stable/reference/generators.html
        """
        pass

    def _convert_labels_to_int(self) -> None:
        """
        Performs two operations:
        1. Convert the labels of the graph to integers - useful for grid like graphs where labels are tuples
        2. Convert the graph to use 1-based indexing - useful for indexing because we reserve 0 for the super node
        """
        if self.graph is None:
            raise ValueError("Graph not initialized")
        self.graph = nx.convert_node_labels_to_integers(self.graph, first_label=1)  # type: ignore

    def initialize(self) -> None:
        """
        Initialize the graph
        """
        self.generate_graph()
        self._convert_labels_to_int()

    def get_all_neighbours(self) -> List[int]:
        """
        Returns the list of neighbours of the current node
        """
        # get all neighbours of the current node using the self.graph attribute
        if self.graph is None:
            raise ValueError("Graph not initialized")
        return list(self.graph.neighbors(self.rank)) # type: ignore

    def sample_neighbours(self, k: int) -> List[int]:
        """
        Returns a random sample of k neighbours of the current node
        If the number of neighbours is less than k, return all neighbours
        """
        if self.graph is None:
            raise ValueError("Graph not initialized")
        neighbours = self.get_all_neighbours()
        if len(neighbours) <= k:
            return neighbours
        return np.random.choice(neighbours, size=k, replace=False).tolist()

    def get_neighbourhood_size(self) -> int:
        """
        Returns the size of the neighbourhood of the current node
        """
        if self.graph is None:
            raise ValueError("Graph not initialized")
        return len(self.get_all_neighbours())
