import numpy as np
import math


class GridTopology:
    def get_selected_ids(node_id, config):
        grid_size = int(config["num_clients"] ** 0.5)

        num_clients = config["num_clients"]

        selected_ids = []

        # Left
        if node_id % grid_size != 1:
            selected_ids.append(node_id - 1)

        # Right
        if node_id % grid_size != 0 and node_id < num_clients:
            selected_ids.append(node_id + 1)

        # Top
        if node_id > grid_size:
            selected_ids.append(node_id - grid_size)

        # Bottom
        if node_id <= num_clients - grid_size:
            selected_ids.append(node_id + grid_size)

        num_clients_to_select = config["num_clients_to_select"]
        # Force self node id to be selected, not removed before sampling to
        # keep sampling identic across nodes (if same seed)
        selected_collabs = np.random.choice(
            selected_ids,
            size=min(num_clients_to_select, len(selected_ids)),
            replace=False,
        )
        selected_ids = list(selected_collabs) + [node_id]

        print("Selected collabs:" + str(selected_ids))

        return selected_ids
