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
        self.neighbor_sample_generator = np.random.default_rng(seed=int(self.config["seed"])*10000 + self.rank ) # type: ignore

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

        # if graph is not fully conncted, print warning in red
        if not nx.is_connected(self.graph):
            print("\033[91m" + "Warning: Graph is not fully connected" + "\033[0m")

    def get_all_neighbours(self) -> List[int]:
        """
        Returns the list of neighbours of the current node
        """
        # get all neighbours of the current node using the self.graph attribute
        # NOTE: graph is 1-indexed, but our node IDs are 0-indexed
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
        return self.neighbor_sample_generator.choice(neighbours, size=k, replace=False).tolist()

    def get_neighbourhood_size(self) -> int:
        """
        Returns the size of the neighbourhood of the current node
        """
        if self.graph is None:
            raise ValueError("Graph not initialized")
        return len(self.get_all_neighbours())
    
    def calculate_graph_metrics(self, target_node:int) -> dict[str, int | dict[int, int]]:
        G = self.graph
        # Calculate graph density
        density = nx.density(G)
        print(f"Graph Density: {density}")

        # Calculate shortest path lengths from every node to the target node
        shortest_paths = nx.shortest_path_length(G, target=target_node)
        print(f"Shortest Path Lengths to Target Node {target_node}: {shortest_paths}")

        # Calculate centrality metrics
        eigenvector_centrality = nx.eigenvector_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        degree_centrality = nx.degree_centrality(G)

        print("Eigenvector Centrality:", eigenvector_centrality)
        print("Closeness Centrality:", closeness_centrality)
        print("Degree Centrality:", degree_centrality)

        return {
            "density": density,
            "shortest_paths": shortest_paths,
            "eigenvector_centrality": eigenvector_centrality,
            "closeness_centrality": closeness_centrality,
            "degree_centrality": degree_centrality
        }
