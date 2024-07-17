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
    "exp_id": "test_noniid3",
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


feddatarepr = {
    "algo": "feddatarepr",
    "exp_id": "test5",
    "num_rep": 1,
    "load_existing": False,

     # Dataset params 
    #"test_samples_per_class": 100,
    #"support":get_sliding_window_support(num_clients=feddatarepr_clients, num_classes=10, num_classes_per_client=4),
    
    # Similarity params
    "representation": "train_data", # "test_data", "train_data", "dreams"
    "num_repr_samples": 16,
    # "CTLR_KL" Collaborator is Teacher using Learner Representation 
    # "CTCR_KL" Collaborator is Teacher using Collaborator Representation - Default row
    # "LTLR_KL" Collaborator is Learner using Learner Representation - Default column
    # "CTAR_KL" Collaborator is Teacher using ALL Representations (from every other client)
    # "train_loss_inv" : 1-loss/total
    # "train_loss_sm": 1-softmax(losses) 
    "similarity_metric": "train_loss_inv",
    
    # Memory params
    "sim_running_average": 10,
    "sim_exclude_first": (5, 5), # (first rounds, first rounds after T0)
    
    # Clients selection
    "target_clients_before_T_0": 0, #feddatarepr_clients-1,
    "target_clients_after_T_0": 1,
    "T_0": 10,   # round after wich only target_clients_after_T_0 peers are kept
    # highest, lowest, [lower_exp]_sim_sampling, top_x, xth, uniform_rdm
    "selection_strategy": "uniform_rdm",#"uniform_rdm", 
    #"eps_greedy": 0.1,
    # "num_clients_top_x" : 1, # Ideally: size community-1
    # "selection_temperature": 0.5, # For all strategy with temperature
    
    
    # Consensus params
    # "sim_averaging", "sim_of_sim", "vote_1hop", "affinity_propagation_clustering", "mean_shift_clustering", "club"
    "consensus":"mean_shift_clustering",# "affinity_propagation_clustering",
    # "affinity_precomputed": False, # If False similarity row are treated as data points and not as similarity values    
    # "club_weak_link_strategy": "own_cluster_and_pointing_to", #"own_cluster_and_pointing_to", pointing_to, own_cluster
    # "vote_consensus": (2,2), #( num_voter, num_vote_per_voter)
    # "sim_consensus_top_a": 3,

    #"community_type": "dataset", 
    #"num_communities": len(domainnet_classes),
    
    # Learning setup
    "warmup_epochs": 5,
    "epochs_per_round": 5,
    "rounds_per_selection": 1, # Number of rounds before selecting new collaborator(s)
    "rounds": 210, 
    "model": "resnet10",
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # Dreams params
    # "reprs_position": 0,
    # "inp_shape": [3, 32, 32] , 
    # "inv_lr": 1e-1, 
    # "inv_epochs": 500, 
    # "alpha_preds": 0.1, 
    # "alpha_tv": 2.5e-3, 
    # "alpha_l2": 1e-7, 
    # "alpha_f": 10.0,
    #"dreams_keep_best": False, # Use reprs with lowest loss 
    
    "exp_keys": ["similarity_metric", "selection_strategy", "consensus"]
}

current_config = fedweight
