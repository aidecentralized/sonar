import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.base_class import BaseFedAvgClient, BaseFedAvgServer
from utils.stats_utils import from_round_stats_per_round_per_client_to_dict_arrays
from torch.utils.data import DataLoader, Dataset

kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

class DistCorrelation(nn.Module):
    def __init__(self, ):
        super(DistCorrelation, self).__init__()

    def pairwise_distances(self, x):
        '''Taken from: https://discuss.pytorch.org/t/batched-pairwise-distance/39611'''
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist[dist != dist] = 0  # replace nan values with 0
        return torch.clamp(dist, 0.0, np.inf)

    def forward(self, z, data):
        z = z.reshape(z.shape[0], -1)
        data = data.reshape(data.shape[0], -1)
        a = self.pairwise_distances(z)
        b = self.pairwise_distances(data)
        a_centered = a - a.mean(dim=0).unsqueeze(1) - a.mean(dim=1) + a.mean()
        b_centered = b - b.mean(dim=0).unsqueeze(1) - b.mean(dim=1) + b.mean()
        dCOVab = torch.sqrt(torch.sum(a_centered * b_centered) / a.shape[1]**2)
        var_aa = torch.sqrt(torch.sum(a_centered * a_centered) / a.shape[1]**2)
        var_bb = torch.sqrt(torch.sum(b_centered * b_centered) / a.shape[1]**2)

        dCORab = dCOVab / torch.sqrt(var_aa * var_bb)
        return dCORab

class CommProtocol(object):
    """
    Communication protocol tags for the server and clients
    """
    ROUND_START = 0 # Server signals the start of a round
    REP1_ADVERT = 1 # Clients advertise their representations with the server
    REPS1_SHARE = 2 # Server shares representations with clients
    REP2_ADVERT = 3 # Clients share their KL divergence with the server
    REPS2_SHARE = 4 # Server shares KL divergence with clients
    C_SELECTION = 5 # Clients send their selected collaborators to the server
    KNLDG_SHARE = 6 # Server shares selected knowledge with clients
    ROUND_STATS = 7 # Clients send their stats to the server
    SIM_ADVERT = 8 # Clients advertise their similarity with the server
    SIM_SHARE = 9 # Server shares similarity with clients
    CONS_ADVERT = 10 # Clients advertise their consensus with the server
    CONS_SHARE = 11 # Server shares consensus with clients


TWO_STEP_STRAT= ["CTAR_KL", "euclidean_pairwise_KL", "LTLR_KL", "train_loss", "CTLR_KL", "log_all_metrics", "dist_corr_AR", "dist_corr_LR"]
CONS_STEP = ["vote_1hop"]

class FedDataRepClient(BaseFedAvgClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.tag = CommProtocol
        
        self.require_second_step = config["similarity_metric"] in TWO_STEP_STRAT
        self.with_sim_consensus = self.config.get("with_sim_consensus", False)
        
        if self.config.get("sim_running_average", 0) > 0:
            self.running_average = {}

        self.set_reprs_parameters()
        
    def set_reprs_parameters(self):
        if self.config["representation"] == "dreams":
            self.inv_optimizer = torch.optim.Adam
            self.loss_r_feature_layers = []
            self.EPS = 1e-8
            
        
        self.lowest_inv_loss = np.inf
        self.lowest_inv_loss_inputs = None

    def get_representation(self):
        reprs = None
        out = None
        reprs_y = None
        
        self.model.eval()
        num_repr_samples = self.config["num_repr_samples"]

        if self.config["representation"] == "train_data":
            reprs_loader = self.dloader
        elif self.config["representation"] == "test_data":
            reprs_loader = self._test_loader
        else:
            raise ValueError("Representation {} not implemented".format(self.config["representation"]))
            
        num_repr_samples = self.config["num_repr_samples"]
        for data, labels in reprs_loader:
                
            # Iterate over the whole dataloader to to leave it in a consistent state
            if reprs is not None and reprs.size(0) >= num_repr_samples:
                continue
            
            with torch.no_grad():
                batch_x = data.to(self.device)

            batch_out = self.model(batch_x)
            
            reprs = batch_x if reprs is None else torch.cat((reprs, batch_x), dim=0)
            out = batch_out if out is None else torch.cat((out, batch_out), dim=0)
            reprs_y = labels if reprs_y is None else torch.cat((reprs_y, labels), dim=0)

        reprs = reprs[:num_repr_samples]
        out = out[:num_repr_samples]
        self.reprs_y = reprs_y[:num_repr_samples]
        reprs = reprs.detach().cpu()
        out = F.log_softmax(out, dim=1).detach().cpu()
         
        self.model.train()
         
        return reprs, out
        
    def get_second_step_representation(self, reprs_dict):
        if not self.require_second_step:
            return None
    
        if self.config["similarity_metric"] in ["CTAR_KL", "train_loss", "CTLR_KL", "log_all_metrics", "dist_corr_AR", "dist_corr_LR"]:
            return self.compute_softlabels(reprs_dict)
        elif self.config["similarity_metric"] in ["euclidean_pairwise_KL", "LTLR_KL"]:
            # LTLR is contained in CTCR from the point of view of the client receiving this representation
            return self.compute_CTCR_KL(reprs_dict)
        else:
            raise ValueError("Similarity metric {} not implemented".format(self.config["similarity_metric"]))
     
    def flatten_repr(self,repr):
        params = []
        
        for key in repr.keys():
            params.append(repr[key].view(-1))

        params = torch.cat(params)
        
        return params
              
    def compute_pseudo_grad_norm(self, prev_wts, new_wts):
        return np.linalg.norm(self.flatten_repr(prev_wts) -  self.flatten_repr(new_wts))

    # === First representation step === #
        
    def compute_softlabels(self, reprs_dict):
        softlabels_dict = {}
        self.model.eval()
        for key, (data, softlabel) in reprs_dict.items():
            if key == self.node_id: 
                softlabels_dict[key] = softlabel
                continue
            
            loader = DataLoader(data, batch_size=self.config["batch_size"], shuffle=False)
            out = None
            for batch in loader:
                with torch.no_grad():
                    batch_out = self.model(batch.to(self.device), position=self.config.get("reprs_position", 0))
                out = batch_out if out is None else torch.cat((out, batch_out), dim=0)
                    
            out = F.log_softmax(out, dim=1).detach().cpu()    
            
            softlabels_dict[key] = out  
        
        self.model.train()

        return softlabels_dict
        
    def compute_CTCR_KL(self, reprs_dict):
        softlabels_dict = self.compute_softlabels(reprs_dict)

        KL_dict = {}
        for client_id, own_softlabels in softlabels_dict.items():
            data, softlabels = reprs_dict[client_id]
            # ,
            # loss_pointwise = kl_loss_pw_fn(own_softlabels, softlabels).sum(dim=1)
            # print("loss_pointwise shape", loss_pointwise.shape)
            # KL_dict[client_id] = (loss_pointwise.sum() / data.size(0)).item()
            KL_dict[client_id] = kl_loss_fn(own_softlabels, softlabels).item()
            
            self.log_clients_stats(KL_dict, "KL divergence CTCR")
        
        return KL_dict
          
    def compute_euclidean(self, reprs_dict):
        softlabels_dict = self.compute_softlabels(reprs_dict)
        
        dist_dict = {}
        for client_id, own_softlabels in softlabels_dict.items():
            _, softlabels = reprs_dict[client_id]
           
            dist_dict[client_id] = ((own_softlabels - softlabels) ** 2).sum().item()
            
            self.log_clients_stats(dist_dict, "Euclidean distance")
        
        return dist_dict
        
    # === Second representation step === #
              
    def compute_CTAR_KL(self, second_step_reprs_dict):
        # Compute the KL divergence between each two clients from their softlabels on every clients' representations
        
        KL_dict = {}
        own_soft_labels = second_step_reprs_dict[self.node_id]
        sorted_reprs_ids = sorted(own_soft_labels.keys())
        cat_softlabels = torch.cat([own_soft_labels[reprs_id] for reprs_id in sorted_reprs_ids], dim=0)
        
        for client_id, client_softlabels  in second_step_reprs_dict.items():
            if client_id == self.node_id:
                KL_dict[self.node_id] = 0
                continue
            
            if set(sorted_reprs_ids) != set(client_softlabels.keys()):
                raise ValueError("Client {} and {} have different keys".format(self.node_id, client_id))
            
            client_cat_softlabels = torch.cat([client_softlabels[reprs_id] for reprs_id in sorted_reprs_ids], dim=0)
            KL_dict[client_id] = kl_loss_fn(cat_softlabels, client_cat_softlabels).item()
            
        self.log_clients_stats(KL_dict, "KL divergence CTAR")

        return KL_dict
     
    def compute_training_loss_similarity(self, second_step_reprs_dict):
        sl_clients_on_own_data = {id: soft_label_dict[self.node_id] for id, soft_label_dict in second_step_reprs_dict.items()}
        sim_dict = {id: self.loss_fn(softlabels, self.reprs_y).item() for id, softlabels in sl_clients_on_own_data.items()}
        
        self.log_clients_stats(sim_dict, "Train loss LR")
        
        total = sum(sim_dict.values())
        sim_dict = {id: 1-v/total for id,v in sim_dict.items()}
        
        return sim_dict
     
    def compute_euclidan_pairwise_KL(self, second_step_reprs_dict):
        clients_ids = sorted(second_step_reprs_dict[self.node_id].keys())
        own_kl = np.array([second_step_reprs_dict[self.node_id][id] for id in clients_ids])
        sim_dict = {}
        for id, kl in second_step_reprs_dict.items():
            if id == self.node_id:
                sim_dict[self.node_id] = 0
                continue
            diff = own_kl - np.array([kl[id] for id in clients_ids])
        
            diff[self.node_id-1] = 0
            diff[id-1] = 0
            
            sim_dict[id] = (diff ** 2).sum()
        
        self.log_clients_stats(sim_dict, "Euclidean distance CTCR KL")
        return sim_dict
             
    def compute_LTLR_KL(self, second_step_reprs_dict):
        # Get the column of the matrix corresponding to the node_id
        sim_dict = {id: kl_dict[self.node_id] for id, kl_dict in second_step_reprs_dict.items()}
        self.log_clients_stats(sim_dict, "KL divergence LTLR")
        return sim_dict
      
    def compute_CTLR_KL(self, second_step_reprs_dict):
        sl_clients_on_own_data = {id: soft_label_dict[self.node_id] for id, soft_label_dict in second_step_reprs_dict.items()}
        own_softlabels = sl_clients_on_own_data[self.node_id]
        sim_dict = {id: 1/max(kl_loss_fn(own_softlabels, softlabels).item(), 1e-10) for id, softlabels in sl_clients_on_own_data.items()}
        
        self.log_clients_stats(sim_dict, "KL divergence CTLR")

        return sim_dict 

    def compute_AR_correlation(self, second_step_reprs_dict):
        loss = DistCorrelation()
        
        corr_dict = {}
        own_soft_labels = second_step_reprs_dict[self.node_id]
        sorted_reprs_ids = sorted(own_soft_labels.keys())
        cat_softlabels = torch.cat([own_soft_labels[reprs_id] for reprs_id in sorted_reprs_ids], dim=0)
        
        for client_id, client_softlabels  in second_step_reprs_dict.items():
           
            if set(sorted_reprs_ids) != set(client_softlabels.keys()):
                raise ValueError("Client {} and {} have different keys".format(self.node_id, client_id))
            
            client_cat_softlabels = torch.cat([client_softlabels[reprs_id] for reprs_id in sorted_reprs_ids], dim=0)
            corr_dict[client_id] = loss(cat_softlabels, client_cat_softlabels).item()
            
        self.log_clients_stats(corr_dict, "Dist Correlation AR")

        return corr_dict
   
    def compute_LR_correlation(self, second_step_reprs_dict):
        loss = DistCorrelation()

        sl_clients_on_own_data = {id: soft_label_dict[self.node_id] for id, soft_label_dict in second_step_reprs_dict.items()}
        own_softlabels = sl_clients_on_own_data[self.node_id]
        sim_dict = {id: loss(own_softlabels, softlabels).item() for id, softlabels in sl_clients_on_own_data.items()}
        
        self.log_clients_stats(sim_dict, "Dist Correlation LR")

        return sim_dict
    
    # === Similarity and client selection === #
       
    def log_all_metrics(self, reprs_dict, second_step_reprs_dict):
        # Computing will automatically log all metrics
        ctar_kl = self.compute_CTAR_KL(second_step_reprs_dict)        
        train_loss = self.compute_training_loss_similarity(second_step_reprs_dict)        
        ctlr_kl = self.compute_CTLR_KL(second_step_reprs_dict)        
        ctcr_kl = self.compute_CTCR_KL(reprs_dict)        
        euclidean = self.compute_euclidean(reprs_dict)
             
    def get_collaborators_similarity(self, round, reprs_dict, second_step_reprs_dict=None):
        if self.config["similarity_metric"] == "CTCR_KL":
            sim_dict = self.compute_CTCR_KL(reprs_dict)
        elif self.config["similarity_metric"] == "euclidean":
            sim_dict = self.compute_euclidean(reprs_dict)
        elif self.config["similarity_metric"] == "train_loss":
            sim_dict = self.compute_training_loss_similarity(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "euclidean_pairwise_KL":
            sim_dict = self.compute_euclidan_pairwise_KL(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "CTAR_KL":
            sim_dict = self.compute_CTAR_KL(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "LTLR_KL":
            sim_dict = self.compute_LTLR_KL(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "CTLR_KL":
            sim_dict = self.compute_CTLR_KL(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "dist_corr_AR":
            sim_dict = self.compute_AR_correlation(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "dist_corr_LR":
            sim_dict = self.compute_LR_correlation(second_step_reprs_dict)
        elif self.config["similarity_metric"] == "log_all_metrics": # Should be combined 
            self.log_all_metrics(reprs_dict, second_step_reprs_dict)
            sim_dict = {self.node_id: 0}
        else:
            raise ValueError("Similarity metric {} not implemented".format(self.config["similarity_metric"]))
        
        num_round_avg = self.config.get("sim_running_average", 0)

        if num_round_avg > 0:
            num_round_exclude, num_round_exclude_after_T0 = self.config.get("sim_exclude_first", (0,0))
            t_0 = self.config.get("T_0", None)
            exclude_round = round < num_round_exclude or (t_0 and t_0 <= round and round < t_0 + num_round_exclude_after_T0)

            if self.node_id == 1:
                print(round, exclude_round)

            for k,v in sim_dict.items():   
                if not exclude_round:
                    if k not in self.running_average:
                        self.running_average[k] = [v]
                    else:
                        self.running_average[k] = self.running_average[k][-num_round_avg:] + [v]
                
                    if self.node_id == 1:
                        print(k, len(self.running_average[k]))
                # If not previous round included return 0
                sim_dict[k] = np.mean(self.running_average.get(k, [0]))
            self.log_clients_stats(sim_dict, "Similarity after running average")
        return sim_dict
                              
    def select_top_k(self, collab_similarity, k, round, total_rounds):
        
        # Remove the nodes that are not in the same community
        collab_similarity = {key: value for key, value in collab_similarity.items() if key in self.communities[self.node_id]}
        
        if self.with_sim_consensus:
            if self.config["similarity_metric"] == "train_loss":
                own_dict = collab_similarity[self.node_id]
                self.log_clients_stats(own_dict, "Similarity before consensus")                

                selected_collab_sim_dict = collab_similarity
                
                # Keep only k highest similarity
                top_a_averaging =self.config.get("sim_consensus_top_a", self.config["num_clients"]-1)
                if top_a_averaging < self.config["num_clients"]-1: 
                    sorted_collab = sorted(own_dict.items(), key=lambda item: item[1], reverse=True)
                    sorted_collab = [(key,v) for key, v in sorted_collab if key != self.node_id][:top_a_averaging]
                    selected_collab_sim_dict = {key: value for key, value in collab_similarity.items() if key in sorted_collab}
                    
                    filtered_own_weights = {key: value for key, value in own_dict.items() if key in sorted_collab}
                    total = sum(filtered_own_weights.values())
                    self.log_clients_stats({key:value/total for key, value in filtered_own_weights.items()}, "Trust weights for consensus")

                    
                new_dict = {id: 0 for id in own_dict.keys()}
                total = 0
                for c_id, c_dict in collab_similarity.items():
                    c_conf = own_dict[c_id]
                    for c1_id, c1_score in c_dict.items():
                        # Does not take client's own similarity into account
                        if c_id == c1_id:
                            new_dict[c1_id] += c_conf * own_dict[c_id]
                        else:
                            new_dict[c1_id] += c_conf * c1_score
                    total+=c_conf
                collab_similarity = {key: v/total for key,v in new_dict.items()}
                
                for key,v in collab_similarity.items():
                    if v > 1:
                        print("Client {} collab {} sim {}".format(self.node_id, key, v))
                
                self.log_clients_stats(collab_similarity, "Consensus similarity")                
            else:
                raise ValueError("Similarity consensus not implemented for {}".format(self.config["similarity_metric"]))

        if k==0 or len(collab_similarity) == 1:
            selected_collab = [self.node_id]
            proba_dist = {self.node_id: 1}
        else:
            strategy = self.config.get("selection_strategy")
            temp = self.config.get("selection_temperature", 1)
            
            if strategy == "highest":
                sorted_collab = sorted(collab_similarity.items(), key=lambda item: item[1], reverse=True)
                selected_collab = [key for key, _ in sorted_collab if key != self.node_id][:k]
                proba_dist = {key: 1 for key in selected_collab}
            elif strategy == "lowest":
                sorted_collab = sorted(collab_similarity.items(), key=lambda item: item[1], reverse=False)
                selected_collab = [key for key, _ in sorted_collab if key != self.node_id][:k]
                proba_dist = {key: 1 for key in selected_collab}
            elif strategy.endswith("sim_sampling"):
                temp = self.config.get("selection_temperature", 1)
                # if strategy.startswith("adaptive_lin"):
                #     temp *= 1 - (round/total_rounds)
                # elif strategy.startswith("adaptive_exp"):
                #     temp *= np.exp(-round/total_rounds)
                
                if strategy == "lower_exp_sim_sampling":
                    proba_dist = {key: np.exp(-value/temp) for key, value in collab_similarity.items()}
                elif strategy == "higher_exp_sim_sampling":                
                    proba_dist = {key: np.exp(value/temp) for key, value in collab_similarity.items()}
                elif strategy == "lower_lin_sim_sampling":
                    total = sum(collab_similarity.values())
                    proba_dist = {key:1 - value/total for key, value in collab_similarity.items()}
                elif strategy == "higher_lin_sim_sampling":         
                    total = sum(collab_similarity.values())
                    proba_dist = {key:value/total for key, value in collab_similarity.items()}           
                else:
                    raise ValueError("Selection strategy {} not implemented".format(strategy))
                proba_dist[self.node_id] = 0
                total = sum(proba_dist.values())
                proba_dist = {key: value/total for key, value in proba_dist.items()}
                items = list(proba_dist.items())
                selected_collab = list(np.random.choice([key for key, _ in items], k, p=[value for _, value in items], replace=False))
            elif strategy.endswith("top_x"):
            #     if not hasattr(self, "growing_top_x"):
            #         self.growing_top_x = 1
            #         self.converged = []
            #         self.prev_pg_norm = {}
                top_x = self.config.get("num_clients_top_x", 0)
                
                
                if strategy == "growing_schedulded_top_x":
                    
                    top_x = 1 + int((self.config["num_clients"]-1) * (round//total_rounds))
                
                # Get the top most similar clients
                sorted_collab = [id for id, value in sorted(collab_similarity.items(), key=lambda item: item[1], reverse=True) if id != self.node_id]
                
                if self.config.get("consensus", None) == "vote_1hop":
                    num_voter = top_x
                    num_vote_per_voter = top_x
                                        
                    voters = sorted_collab[:num_voter]
                    # + 1 to avoid receiving recommendation only about myself
                    self.comm_utils.send_signal(dest=self.server_node, data=sorted_collab[:num_vote_per_voter+1], tag=self.tag.CONS_ADVERT)
                    vote_dict = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.CONS_SHARE)
                    
                    candidate_list = []
                    for voter_id, votes in vote_dict.items():
                        if voter_id in voters:
                            candidate_list += [v for v in votes if v != self.node_id][:num_vote_per_voter]
                    
                    candidate_count = {}
                    for i in candidate_list: 
                        candidate_count[i] = candidate_count.get(i, 0) + 1
                                   
                    self.log_clients_stats(candidate_count, "Vote count")

                    sorted_collab = [id for id, value in sorted(candidate_count.items(), key=lambda item: item[1], reverse=True) if id != self.node_id]
                    
                    
                top_x_collab = [key for key in sorted_collab if key != self.node_id][:top_x]
                
                proba_dist = {key: 1 for key in top_x_collab}
                selected_collab = list(np.random.choice(top_x_collab, k, replace=False))
            else:
                raise ValueError("Selection strategy {} not implemented".format(strategy))
              
            selected_collab.append(self.node_id)
        
        collab_weights = {key: 1/len(selected_collab) if key in selected_collab else 0 for key in collab_similarity.keys()}
        return collab_weights, proba_dist
    
    def log_clients_stats(self, client_dict, stat_name):
        clients_array = np.zeros(self.config["num_clients"])
        for k,v in client_dict.items():
            clients_array[k-1] = v
        self.round_stats[stat_name] = clients_array
        
    def run_protocol(self):
        start_round = self.config.get("start_round", 0)
        total_rounds = self.config["rounds"]
        epochs_per_round = self.config["epochs_per_round"]
        collab_weights_dict = {}
        for round in range(start_round, total_rounds):
            self.round_stats = {}
            
            # Wait on server to start the round
            self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.ROUND_START)
            
            if round == start_round:
                warmup_epochs = self.config.get("warmup_epochs", epochs_per_round)
                if warmup_epochs > 0:
                    warmup_loss, warmup_acc = self.local_train(warmup_epochs)
                    print("Client {} warmup loss {} acc {}".format(self.node_id, warmup_loss, warmup_acc))
            
            repr = self.get_representation()
            self.comm_utils.send_signal(dest=self.server_node, data=repr, tag=self.tag.REP1_ADVERT)
           
            # Collect the representations from all other nodes from the server 
            reprs = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.REPS1_SHARE)
            reprs_dict = {k:v for k,v in enumerate(reprs, 1)}
            
            second_step_reprs_dict = None
            if self.require_second_step:
                reprs2 = self.get_second_step_representation(reprs_dict)
                self.comm_utils.send_signal(dest=self.server_node, data=reprs2, tag=self.tag.REP2_ADVERT)
                second_step_reprs_dict = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.REPS2_SHARE)
                
            similarity_dict = self.get_collaborators_similarity(round, reprs_dict, second_step_reprs_dict)
                        
            if self.with_sim_consensus:
                self.comm_utils.send_signal(dest=self.server_node, data=similarity_dict, tag=self.tag.SIM_ADVERT)
                similarity_dict = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.SIM_SHARE)
                        
            is_num_collab_changed = round == self.config['T_0'] and self.config["target_clients_after_T_0"] != self.config["target_clients_before_T_0"]
            is_selection_round = round == start_round or (round % self.config["rounds_per_selection"] == 0) or is_num_collab_changed
            
            if is_selection_round:
                num_collaborator_before = self.config["target_clients_before_T_0"]
                num_collaborator_after = self.config["target_clients_after_T_0"]
                # num_collaborator = self.config[f"target_clients_{'before' if round < self.config['T_0'] else 'after'}_T_0"]             
                is_before_T0 = round < self.config['T_0']
                if num_collaborator_before == 0 and is_before_T0:
                    # Run selection to log mock selection etc (if not must handle change in server communication)
                    self.select_top_k(similarity_dict, num_collaborator_after, round, total_rounds)
                    collab_weights_dict, proba_dist = {self.node_id: 1}, {self.node_id: 1}
                else:
                    collab_weights_dict, proba_dist = self.select_top_k(similarity_dict, num_collaborator_before if is_before_T0 else num_collaborator_after, round, total_rounds)
                    
                self.log_clients_stats(proba_dist, "Selection probability")
                       
            selected_collaborators = [key for key, value in collab_weights_dict.items() if value > 0]
            self.comm_utils.send_signal(dest=self.server_node, data=(selected_collaborators, self.get_model_weights()), tag=self.tag.C_SELECTION)
            models_wts = self.comm_utils.wait_for_signal(src=self.server_node, tag=self.tag.KNLDG_SHARE)
                                
            avg_wts = self.weighted_aggregate(models_wts, collab_weights_dict, self.model_keys_to_ignore)

            # Average whole model by default
            self.set_model_weights(avg_wts, self.model_keys_to_ignore)
            
            self.round_stats["test_acc_before_training"] = self.local_test()
         
            prev_wts = self.get_model_weights()
            self.round_stats["train_loss"], self.round_stats["train_acc"] = self.local_train(epochs_per_round)
            new_wts = self.get_model_weights()
            self.round_stats["pseudo grad norm"] = self.compute_pseudo_grad_norm(prev_wts, new_wts)

            # Test updated model 
            self.round_stats["test_acc_after_training"] = self.local_test()
                        
            self.log_clients_stats(collab_weights_dict, "Collaborator weights")
            
            self.comm_utils.send_signal(dest=self.server_node, data=self.round_stats, tag=self.tag.ROUND_STATS)

class FedDataRepServer(BaseFedAvgServer):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.set_parameters()
        self.tag = CommProtocol
        self.config = config
        
        self.require_second_step = config["similarity_metric"] in TWO_STEP_STRAT
        self.with_sim_consensus = self.config.get("with_sim_consensus", False)
        
        self.with_cons_step = self.config.get("consensus", "") in CONS_STEP
        print("Server with consensus", self.with_cons_step)

    def send_models_selected(self, collaborator_selection, models_wts):
        for client_id, selected_clients in collaborator_selection.items():
            wts = {key:models_wts[key] for key in selected_clients}
            self.comm_utils.send_signal(dest=client_id, data=wts, tag=self.tag.KNLDG_SHARE)

    def single_round(self):
        """
        Runs the whole training procedure
        """
        
        # Start local training
        self.comm_utils.send_signal_to_all_clients(self.clients, data=None, tag=self.tag.ROUND_START)
        self.log_utils.log_console("Server waiting for all clients to finish local training")
                
        # Collect representation from all clients
        reprs = self.comm_utils.wait_for_all_clients(self.clients, self.tag.REP1_ADVERT)
        self.log_utils.log_console("Server received all clients reprs") 
        
        if self.config["representation"] == "dreams":
            for client, rep in enumerate(reprs):
                # Only store first three channel and 16 images for a 4x4 grid
                imgs = rep[0][:16, :3]
                self.log_utils.log_image(imgs, f"client{client+1}", self.round)
            
        self.comm_utils.send_signal_to_all_clients(self.clients, data=reprs, tag=self.tag.REPS1_SHARE)
        
        if self.require_second_step:
            # Collect the representations from all other nodes from the server 
            reprs2 = self.comm_utils.wait_for_all_clients(self.clients, self.tag.REP2_ADVERT)
            reprs2 = {idx:reprs for idx, reprs in enumerate(reprs2, 1)}
            self.comm_utils.send_signal_to_all_clients(self.clients, data=reprs2, tag=self.tag.REPS2_SHARE)
             
        if self.with_sim_consensus:
            sim_dicts = self.comm_utils.wait_for_all_clients(self.clients, self.tag.SIM_ADVERT)
            sim_dicts = {k:v for k,v in enumerate(sim_dicts, 1)}
            self.comm_utils.send_signal_to_all_clients(self.clients, data=sim_dicts, tag=self.tag.SIM_SHARE)
        
        if self.with_cons_step:            
            consensus = self.comm_utils.wait_for_all_clients(self.clients, self.tag.CONS_ADVERT)
            consensus_dict = {idx:cons for idx, cons in enumerate(consensus, 1)}
            self.comm_utils.send_signal_to_all_clients(self.clients, data=consensus_dict, tag=self.tag.CONS_SHARE)
             
        data = self.comm_utils.wait_for_all_clients(self.clients, self.tag.C_SELECTION)
        collaborator_selection = {idx:select for idx, (select, _) in enumerate(data, 1)}
        models_wts = {idx:model for idx, (_, model) in enumerate(data, 1)}
        self.log_utils.log_console("Server received all clients selection") 

        self.send_models_selected(collaborator_selection, models_wts)
        
        # Collect round stats from all clients
        clients_round_stats = self.comm_utils.wait_for_all_clients(self.clients, self.tag.ROUND_STATS) 
        self.log_utils.log_console("Server received all clients stats") 

        # Log the round stats on tensorboard
        #self.log_utils.log_tb_round_stats(round_stats, ["Collaborator weights", "KL divergence", "selec_probs"], self.round)

        self.log_utils.log_console(f"Round test acc before local training {[stats['test_acc_before_training'] for stats in clients_round_stats]}")
        self.log_utils.log_console(f"Round test acc after local training {[stats['test_acc_after_training'] for stats in clients_round_stats]}")

        return clients_round_stats

    def run_protocol(self):
        self.log_utils.log_console("Starting data repr P2P collaboration")
        start_round = self.config.get("start_round", 0)
        total_round = self.config["rounds"]

        # List of list stats per round
        stats = []
        for round in range(start_round, total_round):
            self.round = round
            self.log_utils.log_console("Starting round {}".format(round))

            round_stats = self.single_round()
            stats.append(round_stats)

        stats_dict = from_round_stats_per_round_per_client_to_dict_arrays(stats)
        stats_dict["round_step"] = 1  
        self.log_utils.log_experiments_stats(stats_dict)
        self.plot_utils.plot_experiments_stats(stats_dict)

        
