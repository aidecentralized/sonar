import math


class Torus:
    def __init__(self, config, node_id):
        self.config = config
        self.node_id = node_id
        self.grid_size = int(math.sqrt(self.config["num_clients"]))
        self.num_clients = self.config["num_clients"]

    def select_collabs(self):
        selected_ids = []
        num_rows = math.ceil(self.num_clients / self.grid_size)

        # Left
        if self.node_id % self.grid_size != 1:
            selected_ids.append(self.node_id - 1)
        elif (
            math.ceil(self.node_id / self.grid_size) * self.grid_size
            <= self.num_clients
        ):
            selected_ids.append(self.node_id + self.grid_size - 1)

        # Right
        if self.node_id % self.grid_size != 0 and self.node_id < self.num_clients:
            right_id = self.node_id + 1
        else:
            node_row = math.ceil(self.node_id / self.grid_size)
            right_id = 1 + self.grid_size * (node_row - 1)

        selected_ids.append(right_id)

        # Top
        if self.node_id > self.grid_size:
            top_id = self.node_id - self.grid_size
        else:
            top_id = self.node_id + self.grid_size * (num_rows - 1)
            if top_id > self.num_clients:
                top_id = top_id - self.grid_size
        selected_ids.append(top_id)

        # Bottom
        if self.node_id <= self.num_clients - self.grid_size:
            bottom_id = self.node_id + self.grid_size
        else:
            bottom_id = self.node_id % self.grid_size
            if bottom_id == 0:
                bottom_id = self.grid_size
        selected_ids.append(bottom_id)

        # Force self node id to be selected, not removed before sampling to keep sampling identical across nodes (if same seed)
        selected_ids = list(set([self.node_id] + selected_ids))

        print("Selected collabs: " + str(self.node_id) + str(selected_ids))


config = {"num_clients": 8}

for node_id in range(1, config["num_clients"] + 1):
    obj = Torus(config, node_id)
    obj.select_collabs()
