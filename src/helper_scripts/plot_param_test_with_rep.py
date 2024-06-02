import numpy as np
from matplotlib import pyplot as plt

FOLDER = "expt_dump/"
RES_FOLDER = f"comparison_plots/"


DATASET = "DomainNet"
SEEDS = []
MAX_ROUNDS = [100]
PARAMS_NAME = "number of samples per clients"
PARAMS_VALUE = [16, 32, 64, 128, 256]#128, 256]#[3, 6, 12]#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
ID_COMP = f"{DATASET}_{PARAMS_NAME}_pre_training".replace(' ', '-')
XLOG = False
TITLE = f"{PARAMS_NAME} comparison, for {DATASET}, 12 clients"
DESCRIPION = f'''
200 rounds, 1 collaborator/round, 12 clients
Shaded area is the standard deviation of {max(len(SEEDS), 1)} runs
'''

exp_folders = []
epx1_runs = []
LEGEND_1 = "Intra domain, pretrained"
for p in PARAMS_VALUE:
    epx1_runs.append({
        "folder":f"domainnet_cli_rea_ske_12clients_{p}spc_fedring_iid_5epr_200r_intra_pretrained{p}_seed1",
        "param_value": p
    })
exp_folders.append((epx1_runs, LEGEND_1))

epx2_runs = []
LEGEND_2 = "Inter domain, pretrained"
for p in PARAMS_VALUE:
    epx2_runs.append({
         "folder":f"domainnet_cli_rea_ske_12clients_{p}spc_fedring_iid_5epr_200r_inter_pretrained{p}_seed1",
        "param_value": p
    })
exp_folders.append((epx2_runs, LEGEND_2))

epx3_runs = []
LEGEND_3 = "Intra domain"
for p in PARAMS_VALUE:
    epx3_runs.append({
        "folder":f"dom_dom_dom_12clients_{p}spc_fedring_iid_5epr_200r_intra_d{p}_seed0",
        "param_value": p
    })
exp_folders.append((epx3_runs, LEGEND_3))

epx4_runs = []
LEGEND_4 = "Inter domain"
for p in PARAMS_VALUE:
    epx4_runs.append({
        "folder":f"dom_dom_dom_12clients_{p}spc_fedring_iid_5epr_200r_inter_d{p}_seed0",
        "param_value": p
    })
    
exp_folders.append((epx4_runs, LEGEND_4))



stats=[
    {
        "file": "test_acc.npy",
        "name": "test accuracy",
        "order": "max",
    },
    {
        "file": "train_acc.npy",
        "name": "train accuracy",
        "order": "max",
    },
    {
        "file": "train_loss.npy",
        "name": "train loss",
        "order": "min",
    },
]

# exp1
#  value1
#   run1
#   run2
#  value2
#   run1
#   run2
# exp2
#  value1
#   ...
   
   
for stat_dict in stats:

    stats_per_exp = []
    min_rounds = np.inf
    for exp, legend in exp_folders:
        stats_per_param_value = []
        for test_value_info in exp:
            runs = []
            if len(SEEDS) == 0:
                path_to_log = FOLDER + test_value_info["folder"] + f"/logs/"
                runs.append(np.load(f"{path_to_log}npy/{stat_dict['file']}"))
                if runs[-1].shape[1] < min_rounds:
                    min_rounds = runs[-1].shape[1]
            else:
                for seed in SEEDS:
                    path_to_log = FOLDER + test_value_info["folder"] + str(seed) + f"/logs/"
                    runs.append(np.load(f"{path_to_log}npy/{stat_dict['file']}"))
                    
                    # Keep track to of the minimum number of rounds so that we can compare all experiments equally
                    if runs[-1].shape[1] < min_rounds:
                        min_rounds = runs[-1].shape[1]
            stats_per_param_value.append(runs)
        stats_per_exp.append((stats_per_param_value, legend))

    if min_rounds == np.inf:
        raise ValueError("No stats found")
    
    if MAX_ROUNDS == []:
        MAX_ROUNDS = [min_rounds]
    
    if max(MAX_ROUNDS) > min_rounds:
        raise ValueError("Sepcified max round is bigger than the minimum number of rounds found in the experiments")
        
    for max_round  in MAX_ROUNDS:
        for stats_per_param_value, legend in stats_per_exp:
            means_for_exp = []
            stds_for_exp = []
            for stats_per_seed in stats_per_param_value:
                bests_for_seed = []
                for stat in stats_per_seed:
                    # Pick one client or mean of all clients ?
                    best = stat[0, :max_round].max() if stat_dict["order"] == "max" else stat[0, :max_round].min()
                    bests_for_seed.append(best)
                
                bests = np.array(bests_for_seed)
                means_for_exp.append(bests.mean())
                stds_for_exp.append(bests.std())
                
            means = np.array(means_for_exp)
            stds = np.array(stds_for_exp)
            
            plt.plot(PARAMS_VALUE, means, marker='*', label=legend)#+ f" {max_round} rounds")
            # fill between quantile
            plt.fill_between(PARAMS_VALUE,
                                means - stds,
                                means + stds,
                                alpha=0.2)
        
    # save figure
    plt.title(TITLE)
    plt.legend()
    if XLOG:
        plt.xscale("log")
    plt.xlabel(f"{PARAMS_NAME}" + DESCRIPION)
    plt.ylabel(stat_dict["name"])
    plt.tight_layout()
    plt.savefig(f"{RES_FOLDER}{ID_COMP}_{stat_dict['name'].replace(' ', '_')}.png")
    plt.close()




