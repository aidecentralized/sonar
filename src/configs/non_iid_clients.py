from utils.config_utils import get_sliding_window_support, get_device_ids
from collections import defaultdict 
# ======================= Supports ======================= #
# With 8 clients not every class is equally represented
sliding_window_8c_4cpc_support = {
    "1": [0, 1, 2, 3],
    "2": [1, 2, 3, 4],
    "3": [2, 3, 4, 5],
    "4": [3, 4, 5, 6],
    "5": [4, 5, 6, 7],
    "6": [5, 6, 7, 8],
    "7": [6, 7, 8, 9],
    "8": [7, 8, 9, 0],
}

support_20c = {
    "1": [0, 1, 2, 3],
    "2": [0, 1, 2, 3],
    "3": [1, 2, 3, 4],
    "4": [1, 2, 3, 4],
    "5": [2, 3, 4, 5],
    "6": [2, 3, 4, 5],
    "7": [3, 4, 5, 6],
    "8": [3, 4, 5, 6],
    "9": [4, 5, 6, 7],
    "10": [4, 5, 6, 7],
    "11": [5, 6, 7, 8],
    "12": [5, 6, 7, 8],
    "13": [6, 7, 8, 9],
    "14": [6, 7, 8, 9],
    "15": [7, 8, 9, 0],
    "16": [7, 8, 9, 0],
    "17": [8, 9, 0, 1],
    "18": [8, 9, 0, 1],
    "19": [9, 0, 1, 2],
    "20": [9, 0, 1, 2],
    }

medmnsit_8c_support = {
    "0": "pathmnist", # Unused in P2P scenarios
    "1": "pathmnist",
    "2": "dermamnist",
    "3": "pathmnist",
    "4": "dermamnist",
    "5": "bloodmnist",
    "6": "tissuemnist",
    "7": "bloodmnist",
    "8": "tissuemnist",
}

medmnsit_4c_support = {
    "0": "pathmnist", # Unused in P2P scenarios
    "1": "pathmnist",
    "2": "dermamnist",
    "3": "pathmnist",
    "4": "dermamnist",
    "5": "bloodmnist",
    "6": "tissuemnist",
    "7": "bloodmnist",
    "8": "tissuemnist",
}

domainnet_4c_support = {
    "0": "domainnet_real", # Unused in P2P scenarios
    "1": "domainnet_sketch",
    "2": "domainnet_real",
    "3": "domainnet_clipart",
}

domainnet_9c_support = {
    "0": "domainnet_real", # Unused in P2P scenarios
    "1": "domainnet_sketch",
    "2": "domainnet_sketch",
    "3": "domainnet_sketch",
    "4": "domainnet_real",
    "5": "domainnet_real",
    "6": "domainnet_real",
    "7": "domainnet_clipart",
    "8": "domainnet_clipart",
    "9": "domainnet_clipart",
}

def get_domain_support(num_clients, base, domains):
    assert num_clients % len(domains) == 0
    
    clients_per_domain = num_clients // len(domains)
    support = {}
    support["0"] = f"{base}_{domains[0]}"
    for i in range(1, num_clients+1):
        support[str(i)] = f"{base}_{domains[(i-1) // clients_per_domain]}" 
    return support
    
CIFAR10_ROT_DMN =["r0", "r90", "r180", "r270"]
def get_cifar10_rot_support(num_clients, domains=CIFAR10_ROT_DMN):
    return get_domain_support(num_clients, "cifar10", domains)

DOMAINNET_DMN = ["real", "sketch", "clipart"]
def get_domainnet_support(num_clients, domains=DOMAINNET_DMN):
    return get_domain_support(num_clients, "domainnet", domains)
  
DOMAINNET_DMN_FULL = ["real", "sketch", "clipart", "infograph", "quickdraw", "painting"]
def get_domainnet_support_full(num_clients, domains=DOMAINNET_DMN_FULL):
    return get_domain_support(num_clients, "domainnet", domains)
    
DOMAINNET_DMN_V2 = ["infograph", "quickdraw", "painting"]
def get_domainnet_support_v2(num_clients, domains=DOMAINNET_DMN_V2):
    return get_domain_support(num_clients, "domainnet", domains)
    
IWILDCAM_DMN = list(range(1, 5)) # 245 possible
def get_iwildcam_support(num_clients, domains=IWILDCAM_DMN):
    return get_domain_support(num_clients, "wilds_iwildcam", domains)

# 2 classes
# 3 domains: 0:116'959, 3:132'052, 4:5'3425 in training set
CAMELYON17_DMN = [0, 3, 4] # + 1, 2 in test set
def get_camelyon17_support(num_clients, domains=CAMELYON17_DMN):
    return get_domain_support(num_clients, "wilds_camelyon17", domains)

# Issue every of the 1139 classes has only 1 sample per domain => how to create in domain test set ?
RXRX1_DMN = [0,1,2,3,4,5] # in train set: 0-6, 11-26, 35-41, 46-48, in test set: 7-10, 27-34, 42-45, 49-50
def get_rxrx1_support(num_clients, domains=RXRX1_DMN):
    return get_domain_support(num_clients, "wilds_rxrx1", domains)

# 0: 17'809, 1: 34'816, 2: 1'582, 3: 20'973, 4:1'641, 5: 42
FMOW_DMN = [0,1,3] 
def get_fmow_support(num_clients, domains=FMOW_DMN):
    return get_domain_support(num_clients, "wilds_fmow", domains)

wilds_dpath = defaultdict(lambda :"imgs")

domainnet_base_dir = "/u/abhi24/matlaberp2/p2p/imgs/domainnet/"
domainnet_dpath = {
    "domainnet_real": domainnet_base_dir, 
    "domainnet_sketch": domainnet_base_dir,
    "domainnet_clipart": domainnet_base_dir,
    "domainnet_infograph": domainnet_base_dir,
    "domainnet_quickdraw": domainnet_base_dir,
    "domainnet_painting": domainnet_base_dir,
}

cifar10_rot_dpath = {
    "cifar10_r0": "./imgs/cifar10",
    "cifar10_r90": "./imgs/cifar10", 
    "cifar10_r180": "./imgs/cifar10",
    "cifar10_r270": "./imgs/cifar10",
}

medmnsit_dpath = {
    "pathmnist": "./imgs/pathmnist",
    "dermamnist": "./imgs/dermamnist",
    "bloodmnist": "./imgs/bloodmnist",
    "tissuemnist": "./imgs/tissuemnist",
}

# ======================= Device IDs ======================= #

device_ids_8c_gpu = {
    "node_0": [5],
    "node_1": [5],
    "node_2": [5],
    "node_3": [2],
    "node_4": [2],
    "node_5": [3],
    "node_6": [3],
    "node_7": [3],
    "node_8": [3]
}

# ======================= Algos Configs ======================= #

# iid training data does not garantee not overlapping samples across nodes

fediso_client = 12
fediso = {
    "seed": 1,
    "algo": "fediso",
    "exp_id": "",
    "num_rep": 1,
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fediso_client, num_client_per_gpu=9, available_gpus=[0,1]),

    # Dataset params 
    "dset": get_domainnet_support(fediso_client), 
    "dpath": domainnet_dpath, 
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support", 
    "test_label_distribution": "iid", # Either "iid" "support", 
    "samples_per_client": 256,
    #"test_samples_per_class": 300,

    #"support": get_sliding_window_support(num_clients=fediso_client, num_classes=10, num_classes_per_client=4), 

    # Clients 
    "num_clients": fediso_client,

    # Learning setup
    "rounds": 200, 
    "epochs_per_round": 5,
    "model": "resnet10",
    "model_lr": 1e-4, 
    "batch_size": 16,

    # params for model
    "position": 0, 
    "inp_shape": [128, 3, 32, 32],

    "exp_keys": []
}

L2C = {
    "seed": 1,
    "algo": "l2c",
    "sharing": "weights", #"weights"
    "exp_id": "",
    "load_existing": False,
    "dset": "cifar10",
    "dpath": "./imgs/cifar10",
    "train_label_distribution": "support", # Either "iid", "shard" "support", 
    "test_label_distribution": "iid", # Either "iid" "support", 
    "validation_prop": 0.05,
    "support" : sliding_window_8c_4cpc_support,
    "samples_per_client": 100,
    "dump_dir": "./expt_dump/",

    # Learning setup
    "num_clients": 8,
    "device_ids": device_ids_8c_gpu,
    #"target_clients_before_T_0": 7,
    "target_clients_after_T_0": 1,
    "T_0": 0,   # round after wich only target_clients_after_T_0 peers are kept
    "alpha_lr": 0.1, 
    "alpha_weight_decay": 0.01,
    
    "epochs_per_round": 5,
    "rounds": 30, 
    "model": "resnet18",
    "average_last_layer": False,
    "model_lr": 1e-4, #0.01, #1e-4, 
    "batch_size": 64,
    "optimizer": "sgd",
    "weight_decay": 5e-4,
    
    # params for model
    "position": 0, 
    "inp_shape": [128, 3, 32, 32],

    "exp_keys": []
}

metaL2C_cifar10 = {
    "seed": 2,
    "algo": "metal2c",
    "sharing": "weights", #"updates"
    "exp_id": "",
    "load_existing": False,
    "dset": "cifar10",
    "dpath": "./imgs/cifar10",
    "train_label_distribution": "support", # Either "iid", "shard" "support", 
    "test_label_distribution": "support", # Either "iid" "support", 
    "validation_prop": 0.05,
    "support" : sliding_window_8c_4cpc_support,
   
    "samples_per_client": 512,
    "dump_dir": "./expt_dump/",

    # Learning setup
    "num_clients": 8,

    "device_ids": device_ids_8c_gpu, 
    #"target_clients_before_T_0": 7,
    "target_clients_after_T_0": 1,
    "T_0": 2,
    "K_0": 0,  # number of peers to keep as neighbors at T_0 (!) inverse that in L2C paper
    "T_0": 250,   # round after wich only K_0 peers are kept
    "alpha_lr": 0.1, 
    "alpha_weight_decay": 0.01,

    "epochs_per_round": 5,
    "rounds": 3, 
    "model": "resnet18",
    "average_last_layer": False,
    "model_lr": 1e-4, 
    "batch_size": 64,
    "optimizer": "sgd",
    "weight_decay": 5e-4,

    # params for model
    "position": 0, 
    "inp_shape": [128, 3, 32, 32],

    "exp_keys": []
}

fedweight = {
    "seed": 1,
    "algo": "fedweight",
    "exp_id": "least_sim", # co, cos, eu
    "load_existing": False,
    "dset": "cifar10",
    "dpath": "./imgs/cifar10",
    "train_label_distribution": "support", # Either "iid", "shard" "support", 
    "test_label_distribution": "iid", # Either "iid" "support", 
    "support" : sliding_window_8c_4cpc_support,
    "collab_weights": "class_overlap", # class_overlap, cosine_similarity, euclidean_distance
    "most_smilar": False,
    "samples_per_client": 100,
    "dump_dir": "./expt_dump/",

    # Learning setup
    "num_clients": 8,
    "device_ids": device_ids_8c_gpu,

    "target_clients_before_T_0": 7,
    "target_clients_after_T_0": 1,
    "T_0": 0,   # round after wich only target_clients_after_T_0 peers are kept
    "epochs_per_round": 5,
    "rounds": 30, 
    "model": "resnet18",
    "average_last_layer":False,
    "model_lr": 1e-4, 
    "batch_size": 64,

    # params for model
    "position": 0, 
    "inp_shape": [128, 3, 32, 32],

    "exp_keys": []
}

fedcentral_client = 3
fedcentral = {
    "seed": 1,
    "algo": "centralized",
    "exp_id": "c2",
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fedcentral_client, num_client_per_gpu=6, available_gpus=[0, 1,2, 3, 4, 5, 6, 7]),

    # Dataset params 
    "dset": get_camelyon17_support(fedcentral_client),
    "dpath": wilds_dpath,
    "train_label_distribution": "iid", # Either "iid", "shard" "support", 
    "test_label_distribution": "iid", # Either "iid" "support",     
    "samples_per_client": 16,
    
    #"support" : get_sliding_window_support(num_clients=fedcentral_client, num_classes=10, num_classes_per_client=4),

    "num_clients": fedcentral_client,
    "central_client": 2,
    "mask_last_layer": False,
    "fine_tune_last_layer": False,
    "epochs_per_round": 5,
    "rounds": 100, 
    "model": "resnet10",
    "model_lr": 1e-4, 
    "batch_size": 16,

    # params for model
    "position": 0, 
    "inp_shape": [128, 3, 32, 32],
    
    "exp_keys": []
}

def assign_colab(clients):
    groups = [3,3,3,4]
    dict = {}
    client = 1
    while client <= clients:
        for size in groups:
            group = []
            for i in range(size):
                group.append(client)
                client += 1
            for c in group:
                dict[c] = group
    return dict

fedass_client= 39
fedass = {
    "seed": 1,
    "algo": "fedass",
    "exp_id": "3334_group",
    "num_rep": 1,
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fedass_client, num_client_per_gpu=10, available_gpus=[0,1,2,3,4,5,6,7]),

    # Dataset params 
    "dset": get_domainnet_support(fedass_client),# get_fmow_support(fedran_client), # get_rxrx1_support(fedran_client), # get_domainnet_support(fedran_client), # get_camelyon17_support(fedran_client), 
    "dpath": domainnet_dpath,
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support", 
    "test_label_distribution": "iid", # Either "iid" "non_iid" "support", 
    "samples_per_client": 32,
    #"test_samples_per_class": 300,
    #"test_samples_per_client": 400, # Only for non_iid test distribution
    # "support" : get_sliding_window_support(num_clients=NUM_CLIENT, num_classes=10, num_classes_per_client=4),

    # Clients selection
    "num_clients": fedass_client,
    "strategy": "fixed", # fixed, direct_expo
    "assigned_collaborators": assign_colab(fedass_client),
    #     {
    #     1: [1, 2],
    #     2: [2, 1],
    #     3: [3, 2, 1]
    # },
    
    # Learning setup
    "rounds": 200, 
    "epochs_per_round": 5,
    "model": "resnet10",
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # params for model
    "position": 0, 
    "exp_keys": ["strategy"]
}

fedval_client= 12
fedval = {
    "seed": 1,
    "algo": "fedval",
    "exp_id": "lowest",
    "num_rep": 1,
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fedval_client, num_client_per_gpu=3, available_gpus=[0,1,2,3]),
    "num_clients": fedval_client,

    # Dataset params 
    "dset": get_camelyon17_support(fedval_client),# get_fmow_support(fedran_client), # get_rxrx1_support(fedran_client), # get_domainnet_support(fedran_client), # get_camelyon17_support(fedran_client), 
    "dpath": wilds_dpath,
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support", 
    "test_label_distribution": "iid", # Either "iid" "non_iid" "support", 
    "samples_per_client": 16,
    #"test_samples_per_class": 300,

    # Clients selection
    "selection_strategy": "lowest", # lowest,
    "target_clients_before_T_0": 1,
    "target_clients_after_T_0": 1,
    "T_0": 400,   # round after wich only target_clients_after_T_0 peers are kept
    "community_type": None,#"dataset",
    # "num_communities": len(cifar10_rotations), #len(domainnet_classes),
    
    # Learning setup
    "rounds": 100, 
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr" : False,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # Knowledge transfer params
    "average_last_layer": True,
    "mask_finetune_last_layer": False,

    # params for model
    "position": 0, 
    "exp_keys": []
}

fedwe_client= 12
fedwe = {
    "seed": 1,
    "algo": "fedweight",
    "exp_id": "sim_consensus_cos_",
    "num_rep": 1,
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fedwe_client, num_client_per_gpu=8, available_gpus=[7,6]),

    # Dataset params 
    "dset": get_domainnet_support(fedwe_client),
    "dpath": domainnet_dpath,
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support", 
    "test_label_distribution": "iid", # Either "iid" "non_iid" "support", 
    "samples_per_client": 32,
    #"test_samples_per_class": 300,
    #"test_samples_per_client": 400, # Only for non_iid test distribution
    # "support" : get_sliding_window_support(num_clients=NUM_CLIENT, num_classes=10, num_classes_per_client=4),

    # Clients selection
    "num_clients": fedwe_client,
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

fedran_client= 39
fedran = {
    "seed": 1,
    "algo": "fedran",
    "exp_id": "warmup_",
    "num_rep": 1,
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fedran_client, num_client_per_gpu=10, available_gpus=[0,1,2,3]),

    # Dataset params 
    "dset": get_domainnet_support(fedran_client),# get_fmow_support(fedran_client), # get_rxrx1_support(fedran_client), # get_domainnet_support(fedran_client), # get_camelyon17_support(fedran_client), 
    "dpath": domainnet_dpath,
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support", 
    "test_label_distribution": "iid", # Either "iid" "non_iid" "support", 
    "samples_per_client": 32,
    #"test_samples_per_class": 300,
    #"test_samples_per_client": 400, # Only for non_iid test distribution
    # "support" : get_sliding_window_support(num_clients=NUM_CLIENT, num_classes=10, num_classes_per_client=4),

    # Clients selection
    "num_clients": fedran_client,
    "target_clients_before_T_0": 0,
    "target_clients_after_T_0": 1,
    "T_0": 10,   # round after wich only target_clients_after_T_0 peers are kept
    "leader_mode": False,
    "community_type": "dataset",
    #"within_community_sampling": 0.1,
    #"p_within_decay": "log_inc", #exp_inc, exp_dec, lin_inc, lin_dec
    #"num_communities": len(cifar10_rotations), #len(domainnet_classes),
    
    # Learning setup
    "rounds": 210, 
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr" : True,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # Knowledge transfer params
    # "inter_commu_layer": "l2", # the layer until which the knowledge is transferred when collaborating outside community (within_community_sampling<1) [l1, l2, l3, l4, fc]
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    #"own_aggr_weight": 0.3,
    # "aggr_weight_strategy": "linear",

    # params for model
    "position": 0, 
    "exp_keys": []
}

# static

fedring_client= 12
fedring = {
    "seed": 1,
    "algo": "fedring",
    "exp_id": "warmup_",
    "num_rep": 1,
    "load_existing": False,
    "dump_dir": "./expt_dump/",
    "device_ids": get_device_ids(num_clients=fedring_client, num_client_per_gpu=10, available_gpus=[0,1,2,3]),

    # Dataset params 
    "dset": get_domainnet_support(fedring_client),# get_fmow_support(fedran_client), # get_rxrx1_support(fedran_client), # get_domainnet_support(fedran_client), # get_camelyon17_support(fedran_client), 
    "dpath": domainnet_dpath,
    "train_label_distribution": "iid", # Either "iid", "non_iid" "support", 
    "test_label_distribution": "iid", # Either "iid" "non_iid" "support", 
    "samples_per_client": 32,
    #"test_samples_per_class": 300,
    #"test_samples_per_client": 400, # Only for non_iid test distribution
    # "support" : get_sliding_window_support(num_clients=NUM_CLIENT, num_classes=10, num_classes_per_client=4),

    # Clients selection
    "num_clients": fedring_client,
    "num_clients_to_select": 1,
    "leader_mode": False,
    "community_type": "dataset",
    #"within_community_sampling": 0.1,
    #"p_within_decay": "log_inc", #exp_inc, exp_dec, lin_inc, lin_dec
    #"num_communities": len(cifar10_rotations), #len(domainnet_classes),
    
    # Learning setup
    "rounds": 210, 
    "epochs_per_round": 5,
    "model": "resnet10",
    "local_train_after_aggr" : True,
    # "pretrained": True,
    # "train_only_fc": True,
    "model_lr": 1e-4, 
    "batch_size": 16,
    
    # Knowledge transfer params
    # "inter_commu_layer": "l2", # the layer until which the knowledge is transferred when collaborating outside community (within_community_sampling<1) [l1, l2, l3, l4, fc]
    "average_last_layer": True,
    "mask_finetune_last_layer": False,
    #"own_aggr_weight": 0.3,
    # "aggr_weight_strategy": "linear",

    # params for model
    "position": 0, 
    "exp_keys": []
}


# current_config = fedcentral

current_config = fedring
# current_config["test_param"] ="community_type"
# current_config["test_values"] = ["dataset", None] 

# __system config__ #
# infra file - device ids etc.
# merging algorithm
# sharing algorithm
# topology file