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
    "exp_type": "iid_clients_federated",
    # Learning setup
    "epochs": 50,
    "model": "resnet34",
    "model_lr": 3e-4,
    "batch_size": 256,
    "exp_keys": [],
}

fedweight = {
    "algo": "fedweight",
    "exp_id": "test2",
    "num_rep": 1,
    "load_existing": False,

    # Dataset params 
    #"test_samples_per_class": 300,
    #"test_samples_per_client": 400, # Only for non_iid test distribution
    # "support" : get_sliding_window_support(num_clients=NUM_CLIENT, num_classes=10, num_classes_per_client=4),

    # Clients selection
    "target_clients": 1,
    "similarity": "CosineSimilarity", #"EuclideanDistance", "CosineSimilarity", 
    #"community_type": "dataset",
    "with_sim_consensus": True,
    
    # Learning setup
    "rounds": 200, 
    "epochs_per_round": 5,
    "warmup_epochs": 50,
    "model": "resnet10",
    "local_train_after_aggr" : True,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    #"own_aggr_weight": 0.3,
    #"aggr_weight_strategy": "linear",

    # params for model
    "position": 0, 
    "exp_keys": []
}

current_config = fedweight
