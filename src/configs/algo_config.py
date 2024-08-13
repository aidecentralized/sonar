# Algorithm Configuration

iid_dispfl_clients_new = {
    "algo": "dispfl",
    "exp_id": 200,
    "exp_type": "iid_dispfl",
    "neighbors": 2,
    "active_rate": 0.8,
    "dense_ratio": 0.5,
    "erk_power_scale": 1,
    "anneal_factor": 0.5,
    "epochs": 1000,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 128,
    "exp_keys": []
}

traditional_fl = {
    "algo": "fedavg",
    "exp_id": 10,
    "exp_type": "test_object_detect3",
    # Learning setup
    "epochs": 50,
    "model": "yolo",
    "model_lr": 1e-5,
    "batch_size": 32,
    "exp_keys": [],
}

current_config = traditional_fl
