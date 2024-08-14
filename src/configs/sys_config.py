# System Configuration
# TODO: Set up multiple non-iid configurations here. The goal of a separate system config
# is to simulate different real-world scenarios without changing the algorithm configuration.
from typing import Dict, List
import socket


mpi_system_config = {
    "comm": {
        "type": "MPI"
    },
    "num_users": 4,
    # "experiment_path": "./experiments/",
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 31,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [0], "node_1": [0],"node_2": [0], "node_3": [0]},
    "samples_per_user": 1000, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

object_detect_system_config = {
    "num_users": 1,
    "experiment_path": "./experiments/",
    "dset": "pascal",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/pascal/VOCdevkit/VOC2012/",
    "seed": 37,
    # node_0 is a server currently
    # The device_ids dictionary depicts the GPUs on which the nodes reside.
    # For a single-GPU environment, the config will look as follows (as it follows a 0-based indexing):
    "device_ids": {"node_0": [1], "node_1": [2]},
    "samples_per_user": 100, #TODO: To model scenarios where different users have different number of samples
    # we need to make this a dictionary with user_id as key and number of samples as value
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

def is_port_available(port: int) -> bool:
    """
    Check if a port is available for use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # type: ignore
        return s.connect_ex(('localhost', port)) != 0 # type: ignore


def generate_ports(num_users: int) -> List[int]:
    """
    Generate a list of ports that are available for use.
    """
    ports: List[int] = []
    i = 0
    while len(ports) < num_users:
        port = 50051 + i
        # check if the port is available
        if is_port_available(port):
            ports.append(port)
        else:
            print(f"Port {port} is not available, skipping...")
        i += 1
    return ports

def generate_peer_ids(num_users: int) -> List[str]:
    """
    Generate a list of peer IDs for the users.
    """
    peer_ids: List[str] = []
    ports = generate_ports(num_users)
    for i in range(num_users):
        peer_ids.append(f"localhost:{ports[i]}")
    return peer_ids

def get_device_ids(num_users: int, gpus_available: List[int]) -> Dict[str, List[int]]:
    """
    Get the GPU device IDs for the users.
    """
    device_ids: Dict[str, List[int]] = {}
    for i in range(num_users):
        index = i % len(gpus_available)
        gpu_id = gpus_available[index]
        device_ids[f"node_{i}"] = [gpu_id]
    return device_ids

num_users = 50
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
grpc_system_config = {
    "num_users": num_users,
    "comm": {
        "type": "GRPC",
        "all_peer_ids": generate_peer_ids(num_users + 1) # +1 for the super-node,
    },
    "dset": "cifar10",
    "dump_dir": "./expt_dump/",
    "dpath": "./datasets/imgs/cifar10/",
    "seed": 2,
    "device_ids": get_device_ids(num_users + 1, gpu_ids), # +1 for the super-node
    "samples_per_user": 500,
    "train_label_distribution": "iid",
    "test_label_distribution": "iid",
    "folder_deletion_signal_path":"./expt_dump/folder_deletion.signal"
}

current_config = grpc_system_config
