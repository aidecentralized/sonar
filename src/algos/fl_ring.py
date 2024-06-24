class RingTopology():
    def get_selected_ids(node_id, config):
        if((node_id + 1) % config["num_clients"] == 0):
            selected_ids = [(node_id + 2) % config["num_clients"]]
        else:
            selected_ids = [(node_id + 1) % config["num_clients"]]

        num_clients_to_select = config["num_clients_to_select"]

        # Force self node id to be selected, not removed before sampling to keep sampling identic across nodes (if same seed)
        selected_ids = [node_id] + [id for id in selected_ids if id != node_id][:num_clients_to_select]
        return selected_ids