from typing import Any, Dict, List, Union
from configs.sys_config import SystemConfig
import jmespath
import importlib

# Updated to return SystemConfig if defined as a class instance
def load_config(config_path: str) -> Union[Dict[str, Any], Any]:
    path = ".".join(config_path.split(".")[1].split("/")[1:])
    config_module = importlib.import_module(path)

    # Check if current_config is an instance of SystemConfig and return directly
    config = getattr(config_module, "current_config", None)
    if isinstance(config, SystemConfig):
        return config
    else:
        raise TypeError("Expected current_config to be an instance of SystemConfig.")


def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # Set default values if missing in the config
    config.setdefault("num_gpus", len(config.get("device_ids", [0])))
    config.setdefault("batch_size", 64)
    config.setdefault("seed", 1)
    config.setdefault("samples_per_user", 1000)  # Default samples per user if missing
    config.setdefault("dset", "default_dataset_name")  # Default dataset

    # Create dataset name and experiment name based on config values
    if isinstance(config["dset"], dict):
        dsets = sorted(set(config["dset"].values()))
        dset = "_".join([d.split("_")[0] for d in dsets]) if len(dsets) > 1 else dsets[0]
    else:
        dset = config["dset"]

    experiment_name = f"{dset}_{config['num_users']}users_{config['samples_per_user']}_{config.get('exp_id', 'exp')}"
    
    for exp_key in config.get("exp_keys", []):
        item = jmespath.search(exp_key, config)
        if item is not None:
            experiment_name += f"_{item}"

    # Paths for experiment results
    experiments_folder = config.get("dump_dir", "./experiments/")
    results_path = f"{experiments_folder}{experiment_name}_seed{config['seed']}"
    config.update({
        "experiment_name": experiment_name,
        "dset_name": dset,
        "log_path": f"{results_path}/logs/",
        "images_path": f"{results_path}/images/",
        "results_path": results_path,
        "saved_models": f"{results_path}/saved_models/",
        "plot_path": f"{results_path}/plots/"
    })
    
    return config

# No change required for get_sliding_window_support; it still fits the framework
def get_sliding_window_support(num_users: int, num_classes: int, num_classes_per_client: int):
    num_client_with_same_support = max(num_users // num_classes, 1)
    support: Dict[str, List[int]] = {}
    for i in range(1, num_users + 1):
        support[str(i)] = [(((i - 1) // num_client_with_same_support) + j) % num_classes for j in range(num_classes_per_client)]
    return support

# Updated to use a flexible list for GPU assignments
def get_device_ids(num_users: int, num_client_per_gpu: int, available_gpus: List[int]) -> Dict[str, List[int]]:
    assert num_users <= len(available_gpus) * num_client_per_gpu
    device_ids: Dict[str, List[int]] = {}
    gpu_id = 0
    for i in range(1, num_users + 1):
        device_ids[f"node_{i}"] = [available_gpus[gpu_id]]
        gpu_id = (gpu_id + 1) % len(available_gpus)
    device_ids["node_0"] = [available_gpus[0]]
    return device_ids
