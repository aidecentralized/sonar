import numpy as np
import math


class TorusTopology:
    def get_selected_ids(self, node_id, config):
        grid_size = int(math.sqrt(config["num_users"]))
        num_users = config["num_users"]

        print(num_users)

        print(grid_size)

        selected_ids = []
        
        num_rows = math.ceil(num_users / grid_size)

        # Left
        if node_id % grid_size != 1:
            selected_ids.append(node_id - 1)
        elif math.ceil(node_id / grid_size) * grid_size <= num_users:
            selected_ids.append(node_id + grid_size - 1)

        # Right
        if node_id % grid_size != 0 and node_id < num_users:
            right_id = node_id + 1
        else:
            node_row = math.ceil(node_id / grid_size)
            right_id = 1 + grid_size * (node_row - 1)
        selected_ids.append(right_id)

        # Top
        if node_id > grid_size:
            top_id = node_id - grid_size
        else:
            top_id = node_id + grid_size * (num_rows - 1)
            if top_id > num_users:
                top_id = top_id - grid_size
        selected_ids.append(top_id)

        # Bottom
        if node_id <= num_users - grid_size:
            bottom_id = node_id + grid_size
        else:
            bottom_id = node_id % grid_size
            if bottom_id == 0:
                bottom_id = grid_size
        selected_ids.append(bottom_id)

        # Force self node id to be selected, not removed before sampling to
        # keep sampling identical across nodes (if same seed)
        selected_ids = list(set(selected_ids))

        if(num_users == 1):
            selected_ids = [1]
        elif(num_users == 2):
            if(node_id == 1):
                selected_ids = [2]
            else:
                selected_ids = [1]
        elif(num_users == 3):
            if(node_id == 1):
                selected_ids = [2, 3]
            elif(node_id == 2):
                selected_ids = [1]
            else:
                selected_ids = [1]

        num_users_to_select = config["num_users_to_select"]
        selected_collabs = np.random.choice(
            selected_ids,
            size=min(num_users_to_select, len(selected_ids)),
            replace=False,
        )
        selected_ids = list(selected_collabs) + [node_id]

        print("Selected collabs: " + str(node_id) + str(selected_ids))

        return selected_ids
