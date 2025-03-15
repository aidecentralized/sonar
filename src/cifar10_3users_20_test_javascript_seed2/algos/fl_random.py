import random
import numpy as np


class RandomTopology:
    """Method docstring: Returns selected IDs based on some criteria."""

    def get_selected_ids(self, node_id, config, reprs_dict, communities):
        within_community_sampling = config.get("within_community_sampling", 1)

        if random.random() <= within_community_sampling or len(communities) == 1:
            # Consider only neighbors (clients in the same community)
            indices = [
                id
                for id in sorted(list(reprs_dict.keys()))
                if id in communities[node_id]
            ]
        else:
            # Consider clients from other communities
            indices = [
                id
                for id in sorted(list(reprs_dict.keys()))
                if id not in communities[node_id]
            ]

        num_clients_to_select = config[
            f"target_clients_{'before' if round < config['T_0'] else 'after'}_T_0"
        ]
        selected_ids = random.sample(
            indices, min(num_clients_to_select + 1, len(indices))
        )
        # Force self node id to be selected, not removed before sampling to
        # keep sampling identic across nodes (if same seed)
        selected_ids = [node_id] + [id for id in selected_ids if id != node_id][
            :num_clients_to_select
        ]

        return selected_ids
