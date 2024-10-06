from typing import Any, Dict, List
import jmespath
import importlib


def load_config(config_path: str) -> Dict[str, Any]:
    path = ".".join(config_path.split(".")[1].split("/")[1:])
    config = importlib.import_module(path).current_config
    return config


def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config["num_gpus"] = len(config.get("device_ids", [0]))
    config["batch_size"] = config.get("batch_size", 64)  # * config['num_gpus']
    config["seed"] = config.get("seed", 1)
    config["load_existing"] = config.get("load_existing") or False

    if isinstance(config["dset"], dict):
        dset = config.get("dset", {})
        dsets = list(set(dset.values()))
        dsets = sorted(dsets)
        if min([len(d.split("_")) for d in dsets]) <= 1:
            dset = "_".join([d[:3] for d in dsets])
        else:
            dset = f"{dsets[0].split('_')[-2]}_" + "_".join(
                [d.split("_")[-1][:3] for d in dsets]
            )
    else:
        dset = config["dset"]

    experiment_name = "{}_{}users_{}_{}".format(
        dset,
        config["num_users"],
        config["samples_per_user"],
        config["exp_id"],
    )

    for exp_key in config["exp_keys"]:
        item = jmespath.search(exp_key, config)
        assert item is not None
        key = exp_key.split(".")[-1]
        assert key is not None
        # experiment_name += "_{}_{}".format(key, item)
        experiment_name += "_{}".format(item)

    experiments_folder = config["dump_dir"]
    results_path = experiments_folder + experiment_name + f"_seed{config['seed']}"

    log_path = results_path + "/logs/"
    images_path = results_path + "/images/"
    plot_path = results_path + "/plots/"

    config["experiment_name"] = experiment_name
    config["dset_name"] = dset
    config["log_path"] = log_path
    config["images_path"] = images_path
    config["results_path"] = results_path
    config["saved_models"] = results_path + "/saved_models/"
    config["plot_path"] = plot_path

    return config


def get_sliding_window_support(
    num_users: int, num_classes: int, num_classes_per_client: int
):
    num_client_with_same_support = max(num_users // num_classes, 1)
    support: Dict[str, List[int]] = {}
    # Slide window by 1, clients with same support are consecutive
    for i in range(1, num_users + 1):
        support[str(i)] = [
            (((i - 1) // num_client_with_same_support) + j) % num_classes
            for j in range(num_classes_per_client)
        ]
    return support


def get_device_ids(num_users: int, num_client_per_gpu: int, available_gpus: list[int]):
    assert num_users <= len(available_gpus) * num_client_per_gpu
    device_ids: Dict[str, List[int]] = {}

    gpu_id = 0
    for i in range(1, num_users + 1):
        device_ids[f"node_{i}"] = [available_gpus[gpu_id]]
        gpu_id = (gpu_id + 1) % len(available_gpus)  # Alternate GPU assignment

    # Assign the server to the first GPU
    device_ids[f"node_0"] = [available_gpus[0]]

    return device_ids
