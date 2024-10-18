import random
from utils.types import ConfigType

class NodeDropout:
    """
    A class that manages the participation of a node.
    
    Attributes:
        dropout_rate (float): The probability that a node will drop out.
        dropout_correlation (float): The correlation between the dropout of a node and its neighbors.
        sample (lambda): A function that samples a real number based of dropout distribution.

    """
    
    def __init__(self, node_id: int, dropout_dict: dict, rng: random.Random) -> None:
        """
        Initializes the NodeDropout class with the provided dropout rate and correlation.

        Args:
            dropout_rate (float): The probability that a node will drop out.
            dropout_correlation (float): The correlation between the dropout of a node and its neighbors.
        """
        self.node_id = node_id
        
        self.dropout_enabled = len(dropout_dict) > 0 and float(dropout_dict.get("dropout_rate", 0.0)) > 0.0
        
        if self.dropout_enabled:
            self.dropout_configs = dropout_dict
            self.dropout_rate = float(dropout_dict.get("dropout_rate", 0))
            self.dropout_correlation = float(dropout_dict.get("dropout_correlation", 0))
            dropout_distribution_dict = dropout_dict.get("dropout_distribution_dict", {})
            dropout_method_params = dropout_distribution_dict.get("parameters", {})
            dropout_method = dropout_distribution_dict.get("method", "uniform")
            self.rng = rng
            self.dropped_recently = False


            if dropout_method == "uniform":
                self.sample = lambda: self.rng.random()
            elif dropout_method == "normal":
                assert ("mean" in dropout_method_params), "mean must be provided for normal dropout distribution"
                assert ("std" in dropout_method_params), "std must be provided for normal dropout distribution"
                mean = float(dropout_method_params["mean"])
                std = float(dropout_method_params["std"])
                self.sample = lambda: self.rng.normal(mean, std)
            else:
                raise ValueError(f"Invalid dropout distribution method {dropout_method}! Must be one of uniform | normal")
            
            self.is_available = self.__available
        else:
            self.is_available = lambda: True
            

    def __available(self):
        if self.dropped_recently:
            to_drop = self.sample() < (
                self.dropout_rate
                + self.dropout_correlation * (1 - self.dropout_rate)
            )
        else:
            to_drop = self.sample() < self.dropout_rate
        self.dropped_recently = to_drop
        return not to_drop
