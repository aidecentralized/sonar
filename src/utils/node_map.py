# Making a singleton class, so that only one instance can be created
class NodeMap:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(NodeMap, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'node_map'):
            self.node_map = {}

    def add_node(self, node_id: int, malicious_type: int):
        self.node_map[node_id] = malicious_type

    def get_malicious_type(self, node_id: int) -> int:
        return self.node_map.get(node_id, None)

    def is_malicious(self, node_id: int) -> bool:
        return node_id in self.node_map