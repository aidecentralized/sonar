"""This module defines the RingTopology class for managing ring topologies in federated learning."""
# pylint: disable=R0903
class RingTopology:
    """Manages the nodes in a federated learning ring topology.

    This class provides methods to select the next node(s) for communication
    in a federated learning setup using a ring topology.
    """
    def get_selected_ids(self, node_id, config):
        """Selects the next node(s) in the federated learning ring.

        Args:
            node_id (int): The ID of the current node.
            config (dict): Configuration parameters including 'num_users' and 'num_users_to_select'.

        Returns:
            list: A list of selected node IDs.
        """

        if (node_id + 1) % config["num_users"] == 0:
            selected_ids = [(node_id + 2) % config["num_users"]]
        else:
            selected_ids = [(node_id + 1) % config["num_users"]]

        num_users_to_select = config["num_users_to_select"]

        # Force self node id to be selected, not removed before sampling to
        # keep sampling identical across nodes (if same seed)
        selected_ids = [node_id] + [
            id for id in selected_ids if id != node_id
        ][:num_users_to_select]
        return selected_ids
