import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
from typing import Any, Dict, List
import torch
from fractions import Fraction
import random

from utils.communication.comm_utils import CommunicationManager
from utils.log_utils import LogUtils
from algos.fl import FedAvgClient, FedAvgServer
from algos.fl_static import FedStaticNode, FedStaticServer

import inversefed

def LaplacianGossipMatrix(G):
    max_degree = max([G.degree(node) for node in G.nodes()])
    W = np.eye(G.number_of_nodes()) - 1/max_degree * nx.laplacian_matrix(G).toarray()
    return W

def get_non_attackers_neighbors(G, attackers):
    """
    G : networkx graph
    attackers : list of the nodes considered as attackers
    returns : non repetetive list of the neighbors of the attackers
    """
    return sorted(set(n for attacker in attackers for n in G.neighbors(attacker)).difference(set(attackers)))

def GLS(X, y, cov):
    """
    Returns the generalized least squares estimator b, such as 
    Xb = y + e
    e being a noise of covariance matrix cov
    """
    X_n, X_m = X.shape
    y_m = len(y)
    s_n = len(cov)
    assert s_n == X_n, "Dimension mismatch"
    try:
        inv_cov = np.linalg.inv(cov)
    except Exception as e:
        print("WARNING : The covariance matrix is not invertible, using pseudo inverse instead")
        inv_cov = np.linalg.pinv(cov)
    return np.linalg.inv(X.T@inv_cov@X)@ X.T@inv_cov@y

class ReconstructOptim(): 
    def __init__(self, G, n_iter, attackers, gossip_matrix = LaplacianGossipMatrix, targets_only = False):
        """
        A class to reconstruct the intial values used in a decentralized parallel gd algorithm
        This class depends only on the graph and the attack parameters n_iter and attackers
        It doesn't depend on the actual updates of one particular execution
        G: networkx graph, we require the nodes to be indexed from 0 to n-1
        n_iter: number of gossip iterations n_iter >= 1
        attackers: indices of the attacker nodes
        gossip_matrix: function that returns the gossip matrix of the graph

        same script as https://github.com/AbdellahElmrini/decAttack/tree/master
        """
        self.G = G 
        self.n_iter = n_iter
        self.attackers = attackers
        self.n_attackers = len(attackers)
        self.W = gossip_matrix(self.G)
        self.Wt = torch.tensor(self.W, dtype = torch.float64)
        self.build_knowledge_matrix_dec()

    def build_knowledge_matrix_dec(self, centralized=False):
        """
        Building a simplified knowledge matrix including only the targets as unknowns
        This matrix encodes the system of equations that the attackers receive during the learning
        We assume that the n_a attackers appear in the beginning of the gossip matrix
        returns :
            knowledge_matrix : A matrix of shape m * n, where m =  self.n_iter*len(neighbors), n = number of targets
        """
        if not centralized:
            W = self.W
            att_matrix = []
            n_targets = len(self.W) - self.n_attackers
            for neighbor in get_non_attackers_neighbors(self.G, self.attackers):
                att_matrix.append(np.eye(1,n_targets,neighbor-self.n_attackers)[0]) # Shifting the index of the neighbor to start from 0

            pW_TT = np.identity(n_targets)

            for _ in range(1, self.n_iter):
                pW_TT = W[self.n_attackers:,self.n_attackers: ] @ pW_TT + np.identity((n_targets))
                for neighbor in get_non_attackers_neighbors(self.G, self.attackers):  
                    att_matrix.append(pW_TT[neighbor-self.n_attackers]) # Assuming this neighbor is not an attacker

            self.target_knowledge_matrix = np.array(att_matrix)
            return self.target_knowledge_matrix
        else:
            # Simplify for centralized FL: no gossip matrix, direct aggregation from clients
            n_targets = len(self.W) - self.n_attackers  # Number of clients (non-attackers)
            
            att_matrix = []
            for client in range(n_targets):
                att_matrix.append(np.eye(1, n_targets, client)[0])  # Identity matrix for each client
            
            self.target_knowledge_matrix = np.array(att_matrix)
            return self.target_knowledge_matrix
    def build_cov_target_only(self, sigma):  # NewName : Build_covariance_matrix
        """
        Function to build the covariance matrix of the system of equations received by the attackers 
        The number of columns corresponds to the number of targets in the system
        See the pseudo code at algorithm 6 in the report
        return :
            cov : a matrix of size m * m, where m = self.n_iter*len(neighbors)
        """
        W = self.W
        W_TT = W[self.n_attackers:, self.n_attackers:]
        neighbors = get_non_attackers_neighbors(self.G, self.attackers) 

        m = self.n_iter*len(neighbors)

        cov = np.zeros((m,m)) 
        # We iteratively fill this matrix line by line in a triangular fashion (as it is a symetric matrix)
        i = 0
        
        while i < m:
            for it1 in range(self.n_iter):
                for neighbor1 in neighbors:
                    j = it1*len(neighbors)
                    for it2 in range(it1, self.n_iter):
                        for neighbor2 in neighbors:
                            s=0
                            for t in range(it1+1):
                                s+=np.linalg.matrix_power(W_TT,it1+it2-2*t)[neighbor1, neighbor2]
                            cov[i,j] = sigma**2 * s
                            cov[j,i] = cov[i,j]
                            j += 1
                    i+=1
        return cov



    def reconstruct_GLS_target_only(self, v, X_A, sigma):
        """
        Function to reconstruct the inital gradients from the values received by the attackers after self.n_iter iterations.
        This method uses GLS estimator
        v (nd.array) : vector containing the values received by the attackers (in the order defined by the gossip)
        sigma : (float) : variance  
        returns :
            x_hat : a vector of shape n * v.shape[1], where n is the number of nodes
        """
        cov = self.build_cov_target_only(sigma)
        n_targets = len(self.W) - self.n_attackers
        neighbors = np.array(get_non_attackers_neighbors(self.G, self.attackers))
        n_neighbors = len(neighbors)
        v = v[self.n_attackers:] # v[:self.n_attackers] are the attacker sent updates which are the same as X_A[:self.n_attackers]
        d = v[0].shape[0]
        W_TA = self.Wt[self.n_attackers:, :self.n_attackers]
        W_TT = self.Wt[self.n_attackers:, self.n_attackers:]
        pW_TT = np.identity(n_targets, dtype = np.float64)
        new_v = []
        B_t = np.zeros((n_targets, d), dtype = np.float64)
        for it in range(self.n_iter):
            X_A_t = X_A[it*self.n_attackers:(it+1)*self.n_attackers]
            pW_TT = W_TT @ pW_TT + np.identity((n_targets), dtype = np.float64)
            theta_T_t = v[it*n_neighbors:(it+1)*n_neighbors]
            new_v.extend(theta_T_t-B_t[neighbors-self.n_attackers])
            B_t = W_TT @ B_t + W_TA @ X_A_t
        v = np.array(new_v)
        try:
            return GLS(self.target_knowledge_matrix, v, cov)
        except Exception as e:
            print(e)
            print("Building the knowledge matrix failed")
            raise
    
    def reconstruct_LS_target_only(self, v, X_A):
        """
        Function to reconstruct the inital gradients from the values received by the attackers after self.n_iter iterations.
        This method uses a Least Squares estimator
        v (nd.array) : vector containing the values received by the attackers (in the order defined by the gossip)
        v looks like (X_A^0, \theta_T^{0+), X_A^1, \theta_T^{1+), ..., X_A^T, \theta_T^{T+)}
        where X_A^t are the attacker sent updates at iteration t and \theta_T^{t+)} are the target sent updates at iteration t
        X_A (nd.array) : vector of size n_a*self.n_iter, containing the attacker sent updates at each iteration
        returns :
            x_hat : a vector of shape n_target * v.shape[1], where n_target is the number of target nodes
        """
        # Prepossessing v to adapt to the target only knowledge matrix

        n_targets = len(self.W) - self.n_attackers
        neighbors = np.array(get_non_attackers_neighbors(self.G, self.attackers))
        n_neighbors = len(neighbors)
        v = v[self.n_attackers:] # v[:self.n_attackers] are the attacker sent updates which are the same as X_A[:self.n_attackers]
        d = v[0].shape[0]
        W_TA = self.Wt[self.n_attackers:, :self.n_attackers]
        W_TT = self.Wt[self.n_attackers:, self.n_attackers:]
        #pW_TT = np.identity(n_targets, dtype = np.float32)
        new_v = []
        B_t = np.zeros((n_targets, d), dtype = np.float64)
        for it in range(self.n_iter):
            X_A_t = X_A[it*self.n_attackers:(it+1)*self.n_attackers]
            #pW_TT = W_TT @ pW_TT + np.identity((n_targets), dtype = np.float64)
            theta_T_t = v[it*n_neighbors:(it+1)*n_neighbors]
            new_v.extend(theta_T_t-B_t[neighbors-self.n_attackers])

            B_t = W_TT @ B_t + W_TA @ X_A_t

        v = torch.stack(new_v).numpy()
        
        try:
            return np.linalg.lstsq(self.target_knowledge_matrix, v)[0]
        except Exception as e:
            print(e)
            print("Building the knowledge matrix failed")
            raise

class GradientInversionFedAvgClient(FedAvgClient):
    """
    Implements ground truth for evaluating inversion attack
    """
    def __init__(self, config: Dict[str, Any], node_id: int, comm: CommunicationManager, log: LogUtils):
        super(GradientInversionFedAvgClient, self).__init__(config, node_id, comm, log)
        # get ground truth and labels for evaluation
        self.ground_truth, self.labels = self.extract_ground_truth(num_images=config["num_images"]) # set reconstruction number

        # TODO somehow get the server to access the ground truth and labels for evaluation
        self.comm_utils.send(0, [self.ground_truth, self.labels])

    def extract_ground_truth(self, num_images=10):
        """
        Randomly extract a batch of ground truth images and labels from self.dloader for gradient inversion attacks.
        
        Args:
            num_images (int): Number of images to extract.
            
        Returns:
            ground_truth (torch.Tensor): Tensor containing the extracted ground truth images.
            labels (torch.Tensor): Tensor containing the corresponding labels.
        """
        # Convert the dataset to a list of (image, label) tuples
        data = list(self.dloader.dataset)
        
        # Randomly sample `num_images` images and labels
        sampled_data = random.sample(data, num_images)
        
        # Separate images and labels
        ground_truth = [img for img, label in sampled_data]
        labels = [torch.as_tensor((label,)) for img, label in sampled_data]
        
        # Stack into tensors
        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        
        return ground_truth, labels

class GradientInversionFedAvgServer(FedAvgServer):
    """
    implements gradient inversion attack to reconstruct training images from other nodes
    """
    def __init__(self, config: Dict[str, Any], comm: CommunicationManager, log: LogUtils):
        super(GradientInversionFedAvgServer, self).__init__(config, comm, log)

        #TODO somehow obtain the client's ground truth and labels for evaluation
        self.ground_truth, self.labels = self.obtain_ground_truth() # should be one list per client

    
    def obtain_ground_truth(self):
        """
        Obtain the ground truth images and labels from the clients for evaluation.
        """
        ground_truth, labels = [], []
        client_list = self.comm_utils.receive([i for i in range(self.num_users)])
        # TODO 1) sort the received items 
        # TODO 2) add tag to indicate we are receiving dummy data
        for i in range(len(client_list)):
            ground_truth_i, labels_i = client_list[i][:10], client_list[i][10:]
            ground_truth.append(ground_truth_i)
            labels.append(labels_i)
        return ground_truth, labels
        
    def inverting_gradients_attack(self):
        """
        Setup the inversion attack for the server.

        Based on reconstruction from weight script: 
        https://github.com/JonasGeiping/invertinggradients/blob/1157b61c6704df42c497ab9eb074c75da5204334/Recovery%20from%20Weight%20Updates.ipynb
        """
        setup = inversefed.utils.system_startup()
        if self.dset == "cifar10":
            # TODO figure out whehether we actually have the dm and ds values in our codebase
            dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]

        # extract input parameters (this should be the averaged server-side params after a round of FedAVG)
        input_params_s = self.comm_utils.all_gather() #[clinet1 param, client2 param, ...]
        self.single_round()
        input_params_t = self.comm_utils.all_gather()
        
        # get the param difference for each client [client1 param diff, client2 param diff, ...]
        param_diffs = []

        # Loop over each client's parameters (assumes input_params_s and input_params_t are lists of lists)
        for client_params_s, client_params_t in zip(input_params_s, input_params_t):
            client_param_diff = [
                param_t - param_s  # element-wise difference of the tensors
                for param_s, param_t in zip(client_params_s, client_params_t)
            ]
            param_diffs.append(client_param_diff)

        assert len(param_diffs) == self.num_users == self.ground_truth, "Number of clients does not match number of param differences"
        config = dict(signed=True,
                    boxed=True,
                    cost_fn='sim',
                    indices='def',
                    weights='equal',
                    lr=0.1,
                    optim='adam',
                    restarts=1,
                    max_iterations=8_000,
                    total_variation=1e-6,
                    init='randn',
                    filter='none',
                    lr_decay=True,
                    scoring_choice='loss')

        for client_i in range(self.num_users):
            # TODO assume that client i correspond to order of received params
            ground_truth_i, labels_i, params_i = self.ground_truth[client_i], self.labels[client_i], param_diffs[client_i]

            local_steps = 1 # number of local steps for client training
            local_lr = self.config["model_lr"] # learning rate for client training
            use_updates = False
            rec_machine = inversefed.FedAvgReconstructor(self.model, (dm, ds), local_steps, local_lr, config,
                                             use_updates=use_updates)
            output, stats = rec_machine.reconstruct(params_i, labels_i, img_shape=(3, 32, 32)) # TODO verify img_shape and change it based on dataset
            test_mse = (output.detach() - ground_truth_i).pow(2).mean()
            feat_mse = (self.model(output.detach())- self.model(ground_truth_i)).pow(2).mean()  
            test_psnr = inversefed.metrics.psnr(output, ground_truth_i, factor=1/ds)

            # optional plotting:
            # plot(output)
            # plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
            #         f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");
            return output, test_mse, test_psnr, feat_mse
    def run_protocol(self):
        """
        basically a carbon copy of fl.py's run protocol. Except attack is launched at the end
        """
        self.log_utils.log_console("Starting clients federated averaging")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        for round in range(start_epochs, total_epochs):

            if round == total_epochs - 1:
                self.log_utils.log_console("Launching inversion attack")
                output, test_mse, test_psnr, feat_mse = self.inverting_gradients_attack()
                self.log_utils.log_console("Inversion attack complete")
                self.log_utils.log_summary(
                    f"Round {round} inversion attack complete. Test MSE: {test_mse}, Test PSNR: {test_psnr}, Feature MSE: {feat_mse}"
                )
                # TODO somehow save output?

            self.log_utils.log_console("Starting round {}".format(round))
            self.log_utils.log_summary("Starting round {}".format(round))
            self.single_round()
            self.log_utils.log_console("Server testing the model")
            loss, acc, time_taken = self.test()
            self.log_utils.log_tb(f"test_acc/clients", acc, round)
            self.log_utils.log_tb(f"test_loss/clients", loss, round)
            self.log_utils.log_console(
                "Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(
                    round, acc, loss, time_taken
                )
            )
            # self.log_utils.log_summary("Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(round, acc, loss, time_taken))
            self.log_utils.log_console("Round {} complete".format(round))
            self.log_utils.log_summary(
                "Round {} complete".format(
                    round,
                )
            )
class GradientInversionFedStaticServer(FedStaticServer):
    """
    implements gradient inversion attack to reconstruct training images from other nodes
    can handle colluding neighbors 

    base on method proposed by https://github.com/AbdellahElmrini/decAttack/tree/master'

    reconstruction method uses InvertingGradients by Jonas Geiping: https://github.com/JonasGeiping/invertinggradients

    The order of stacked params is just the keys of attacker / collaborator IDs in ascending order
    """
    def __init__(self, config: Dict[str, Any], G: nx.Graph):
        # construct graph
        # TODO need to recheck this instantiation depend on graph implementation 
        # TODO keep copy of weights at 0th round (when everyone finishes training 
        self.G = G
        self.neighbors = [i for i in range(self.num_users) if i != self.node_id] # for the server, neighbors are all the clients
        self.attackers = [self.node_id] # for the server, the attacker is itself
        self.end_round = config["rounds"]

    def get_model_parameters(self, ids_list: List[int]):
        """
        returns stacked model parameters
        modeled after Decentralized.get_model_params in decAttack codebase

        TODO verify the actual params getting sent
        """
        param_from_collaborators = self.comm_utils.receive(ids_list)
        params = [[] for p in range(self.model.parameters())]

        for i in range(len(self.neighbors)):
            neighbor_id = self.neighbors[i]
            for j, param in enumerate(param_from_collaborators[neighbor_id]):
                params[j].append(param)

            
        for j in range(len(params)):
            params[j] = torch.stack(params[j])

        return params

    def get_node_weights(self):
        """
        helper function that obtains the param updates of attackers
        uses commProtocol to get the params from the nodes

        TODO double check that neighbors include attacking nodes as well
        """

        # Issue for FL where server is the attacker: attacker gradeint is the averaged gradients from neighbors     
        return self.get_model_parameters(self.neighbors), self.get_model_parameters(self.attackers)


    def launch_attack(self):
        """
        Main function for performing inversion attack when the server is the attacker.
        This should happen after running FedAVG for a single round.
        """
        # Build reconstruction class:
        R = ReconstructOptim(self.G, n_iter=1, attackers=self.attackers)

        # Initial parameters (before aggregation)
        neighbor_params0, attacker_params0 = self.get_node_weights()

        # Run a single round of FedAVG to update server's representation
        self.single_round()

        # Get the updated parameters after aggregation
        neighbor_params_i, attacker_params_i = self.get_node_weights()

        # Collect the difference in parameters for attack
        sent_params = []
        attacker_params = []

        # In centralized FL, server is the attacker
        for i in range(len(neighbor_params0)):  # Loop over all clients
            # Calculate the difference between the initial and updated parameters for neighbors (clients)
            sent_params.append(torch.cat([(neighbor_params_i[j][i] - neighbor_params0[j][i]).flatten().detach() for j in range(self.n_params)]).cpu())
        
        # For the server (attacker), compute the difference between its initial and updated parameters
        attacker_params.append(torch.cat([(attacker_params_i[j] - attacker_params0[j]).flatten().detach() for j in range(self.n_params)]).cpu())

        # Use the collected parameters to reconstruct the images
        x_hat = R.reconstruct_LS_target_only(sent_params, attacker_params)

