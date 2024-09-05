import numpy as np
# Removed unused import: math


class GridTopology:
    def get_selected_ids(self, node_id, config):
        """Selects and returns IDs based on the node_id and configuration.

        Args:
            self: Instance of the class.
            node_id: The ID of the current node.
            config: Configuration parameters, including 'num_users'.

        Returns:
            A list of selected IDs.
        """
        grid_size = int(config["num_users"] ** 0.5)
        num_users = config["num_users"]
        # Additional logic to select and return IDs goes here
        selected_ids = []

        # Left
        if node_id % grid_size != 1:
            selected_ids.append(node_id - 1)

        # Right
        if node_id % grid_size != 0 and node_id < num_users:
            selected_ids.append(node_id + 1)

        # Top
        if node_id > grid_size:
            selected_ids.append(node_id - grid_size)

        # Bottom
        if node_id <= num_users - grid_size:
            selected_ids.append(node_id + grid_size)

        num_users_to_select = config["num_users_to_select"]
        # Force self node id to be selected, not removed before sampling to
        # keep sampling identical across nodes (if same seed)
        selected_collabs = np.random.choice(selected_ids, size=num_users_to_select, replace=False)
        selected_ids = list(selected_collabs) + [node_id]

        return selected_ids
