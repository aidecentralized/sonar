# /////////////// Gradient Inversion Helpers ///////////////

import inversefed
import matplotlib.pyplot as plt

import bisect
import torch
import pickle

import numpy as np
import networkx as nx

from typing import Tuple, Dict, Any, List

from collections import defaultdict, OrderedDict
import copy

import pickle

# CIFAR10 hard-coded mean and std
MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

def one_hot_to_int(one_hot_labels):
    """
    Converts one-hot encoded labels to integer labels.

    Parameters:
        one_hot_labels (np.ndarray): A 2D numpy array where each row is a one-hot encoded label.

    Returns:
        np.ndarray: A 1D numpy array of integer labels.
    """

    if isinstance(one_hot_labels, torch.Tensor):
        if one_hot_labels.dim() != 2:
            raise ValueError(f"Input PyTorch tensor must be 2-dimensional but is {one_hot_labels.dim()}-dimensional.")
        if not torch.all((one_hot_labels == 0) | (one_hot_labels == 1)):
            raise ValueError("Input PyTorch tensor must only contain 0s and 1s.")
        return torch.argmax(one_hot_labels, dim=1)

class ParameterTracker:
    """
    Class to track parameter updates for each client in a decentralized learning setting.
    Implements the logic to record parameter updates, calculate parameter differences, and generate parameter tensors.
    Reference: https://github.com/AbdellahElmrini/decAttack
    """
    def __init__(self, node_id: int, 
                 W: torch.Tensor, 
                 num_clients: int, 
                 n_neighbors: int,
                 gia_attackers: List, 
                 base_params: List,
                 base_param_vals: List,
                 total_rounds: int) -> None:
        self.node_id = node_id
        self.W = W
        self.num_clients = num_clients
        self.n_neighbors = n_neighbors
        self.total_rounds = total_rounds
        
        # Initialize params0 and params as lists of lists
        # Each sublist represents a parameter layer across all clients
        # initialize with random base params of the attacker instead: change from decattack 
        self.params0 = [[param for _ in range(self.num_clients)] for param in base_param_vals]
        self.params = [[param for _ in range(self.num_clients)] for param in base_param_vals]
        
        # Track parameter updates as lists
        self.sent_params = []
        self.attacker_params = []
        
        # Track order of clients, maintaining sorted order by node ID
        self.attackers = gia_attackers
        self.base_params = base_params

        # dump file that tracks previous params and params0
        # round_number : [params0, params]
        self.dump_dict = dict()


    def record_update(self, client_id: int, parameters: OrderedDict, round: int, is_attacker: bool = False) -> None:
        """
        Record parameter updates for a specific client in a specific round. Neighbor updates are assmed to come in order
        # TODO currently not equipped to handle client droupouts or new clients
        # TODO if there are multiple attackers, when invoking the record_update, attacker IDs need to be called in order
        """

        assert len(self.params0) == len(self.params) == len(self.base_params), "Length mismatch between params0 and params and base_params"

        for p in range(len(self.params0)):
            if round == 0:
                self.params0[p][client_id] = parameters[self.base_params[p]]
            self.params[p][client_id] = parameters[self.base_params[p]]
        
        # after params and params0 are updated, update the sent_params and attacker_params with difference
        self._calculate_param_diff(client_id, round, is_attacker)

    def _calculate_param_diff(self, client_id: int, round:int, attacker:bool=False) -> torch.Tensor:
        """
        Calculate parameter differences for a specific client across all parameter layers.
        """
        assert len(self.params0) == len(self.params) == len(self.base_params), "Length mismatch between params0 and params"

        if attacker:
            if round == 0:
                # append difference to sent params
                self.sent_params.append(torch.cat([(self.params[j][client_id].cpu() - self.params0[j][client_id].cpu()).flatten().detach() for j in range(len(self.base_params))]).cpu())
            # otherise, append to attacker params
            self.attacker_params.append(torch.cat([(self.params[j][client_id].cpu() - self.params0[j][client_id].cpu()).flatten().detach() for j in range(len(self.base_params))]).cpu())
        
        else:
            # append difference to sent params
            self.sent_params.append(torch.cat([(self.params[j][client_id].cpu() - self.params0[j][client_id].cpu()).flatten().detach() for j in range(len(self.base_params))]).cpu())

    def update_params0(self, round: int) -> None:
        '''
        called at the end of each client receiving iteration to update the params0
        also dumps params and params0 to dump_dict
        '''
        # 

        self.dump_dict[round] = [self.params0] # log the params0 before einsumming

        if round == 0:
            # Iterate through the outer list
            for i in range(len(self.params0)):
                # Find a valid tensor in the current list to determine the shape and device
                valid_tensor = None
                for item in self.params0[i]:
                    if isinstance(item, torch.Tensor):  # Check if it's a tensor
                        valid_tensor = item
                        break
                
                # If a valid tensor is found, replace empty lists with tensors of the same shape
                if valid_tensor is not None:
                    tensor_shape = valid_tensor.shape
                    target_device = valid_tensor.device  # Get the device of the valid tensor
                    self.params0[i] = [
                        torch.empty(tensor_shape, device=target_device) if isinstance(item, list) and not item else item
                        for item in self.params0[i]
                    ]
                
                # Move all tensors in the current list to the same device as the first tensor
                self.params0[i] = [item.to(target_device) for item in self.params0[i]]

                # Stack the tensors in the current list
                self.params0[i] = torch.stack(self.params0[i]).cpu()

        # perform the einsum
        for j in range(len(self.base_params)):
            # Ensure both tensors are of the same type (Float)
            self.params0[j] = self.params0[j].float()  # Convert to Float if needed
            self.W = self.W.float()  # Ensure W is Float
            self.params0[j] = torch.einsum('mn,n...->m...', self.W.cpu(), self.params0[j])
        
        # fedSGD update
        # learning_rate = 3e-4
        # local_rounds = 1
        # for j in range(len(self.base_params)):
        #     self.params0[j] = self.params0[j].float()
        #     self.W = self.W.float()
        #     # Compute the gradient update
        #     gradient_update = torch.einsum('mn,n...->m...', self.W.cpu(), self.params0[j])
        #     # Apply the FedSGD update
        #     self.params0[j] -= learning_rate * gradient_update / local_rounds

    def clear_params(self, round: int) -> None:
        '''
        called at the start of each round to clear the params: dump params into dump_dict with last round's round number
        '''
        assert round > 0, "Round number should be greater than 0"
        assert self.dump_dict[round-1] is not None, "Dump dict for previous round is empty"
        assert len(self.dump_dict[round-1]) == 1, "Dump dict for previous round only contains params0"

        self.dump_dict[round-1] = self.dump_dict[round-1].append(self.params)

        # reset self.params
        self.params = [[[] for _ in range(self.num_clients)] for _ in range(len(self.base_params))]


    def generate_param_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sent_params and attacker_params tensors ensuring complete and ordered updates.
        """
        print(f"DEBUG len of clients: {self.num_clients}")
        print(f"DEBUG len of attackers: {len(self.attackers)}")
        print(f"DEBUG total rounds: {self.total_rounds}")
        print(f"DEBUG n_neighbors: {self.n_neighbors}") # n_neighbors already excludes attackers

        expected_sent_params_len = len(self.attackers) + (self.n_neighbors) * self.total_rounds
        expected_attacker_len = len(self.attackers) * self.total_rounds

        assert len(self.sent_params) == expected_sent_params_len, f"Expected sent_params len {expected_sent_params_len} but got {len(self.sent_params)}"
        assert len(self.attacker_params) == expected_attacker_len, f"Expected attacker params len {expected_attacker_len} but got {len(self.attacker_params)}"

        # Stack tensors to match reference implementation
        return torch.stack(self.sent_params), torch.stack(self.attacker_params)
    
# matrix based gradient disambiguation
# based on privacy attack in decentralized learning by Elmrini et al.
# https://github.com/AbdellahElmrini/decAttack

def LaplacianGossipMatrix(G):
    max_degree = max([G.degree(node) for node in G.nodes()])
    W = np.eye(G.number_of_nodes()) - 1/max_degree * nx.laplacian_matrix(G).toarray()
    return W

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
        self.n_attackers = len(attackers)
        self.W = gossip_matrix(self.G)
        self.attackers = attackers
        self.Wt = torch.tensor(self.W, dtype=torch.float64)
        # Create and store the mapping of original client IDs to matrix indices
        self.client_mapping = self.create_client_mapping()
        self.build_knowledge_matrix_dec()
        self.n_neighbors = len(self.get_non_attackers_neighbors(self.G, self.attackers))

    def get_non_attackers_neighbors(self, G, attackers) -> List[int]:
        """
        G : networkx graph
        attackers : list of the nodes considered as attackers
        returns : non repetetive list of the neighbors of the attackers, sorted
        """
        neighbors = sorted(set(n for attacker in attackers for n in G.neighbors(attacker)).difference(set(attackers)))
        return [n - 1 for n in neighbors]
        # neighbors = set(self.client_mapping["id_to_idx"][n] for attacker in attackers for n in G.neighbors(attacker))
        # return sorted(neighbors.difference(set(attackers)))
    
    def create_client_mapping(self) -> Dict[str, Dict[int, int]]:
        """
        Creates a bidirectional mapping between original client IDs and matrix indices
        Returns:
            Dict containing id_to_idx and idx_to_id mappings
        """
        # Get all nodes from the graph
        all_nodes = set(self.G.nodes())  # Use actual graph nodes instead of range
        non_attackers = sorted(all_nodes.difference(set(self.attackers)))
        
        # Create mappings
        id_to_idx = {client_id: idx for idx, client_id in enumerate(non_attackers)}
        idx_to_id = {idx: client_id for client_id, idx in id_to_idx.items()}
        
        return {
            'id_to_idx': id_to_idx,  # Original client ID -> Matrix index
            'idx_to_id': idx_to_id   # Matrix index -> Original client ID
        }
    
    def get_original_client_id(self, matrix_idx: int) -> int:
        """
        Converts matrix index back to original client ID
        """
        return self.client_mapping['idx_to_id'][matrix_idx]
    
    def build_knowledge_matrix_dec(self, centralized=False):
        print("non_attackers_neighbors: ", self.get_non_attackers_neighbors(self.G, self.attackers))

        if not centralized:
            W = self.W
            att_matrix = []
            n_targets = len(self.W) - self.n_attackers
            print(f"DEBUG: n_targets in build knowledge matrix dec: {n_targets}")
            
            # Get neighbors using original indices
            neighbors = self.get_non_attackers_neighbors(self.G, self.attackers)
            print(f"DEBUG: neighbors in build knowledge matrix dec: {neighbors}")
            print(f"DEBUG: attackers in build knowledge matrix dec: {self.attackers}")
            print(f"DEBUG: n_attackers in build knowledge matrix dec: {self.n_attackers}")

            # Initial matrix using mapped indices
            for neighbor in neighbors:
                att_matrix.append(np.eye(1,n_targets,neighbor-self.n_attackers)[0])

            pW_TT = np.identity(n_targets)

            for _ in range(1, self.n_iter):
                pW_TT = W[self.n_attackers:,self.n_attackers: ] @ pW_TT + np.identity((n_targets))
                # for neighbor in self.get_non_attackers_neighbors(self.G, self.attackers):  
                for neighbor in self.get_non_attackers_neighbors(self.G, self.attackers): 
                    att_matrix.append(pW_TT[neighbor-self.n_attackers]) # Assuming this neighbor is not an attacker


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
        neighbors = self.get_non_attackers_neighbors(self.G, self.attackers) 

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
        neighbors = np.array(self.get_non_attackers_neighbors(self.G, self.attackers))
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
        neighbors = np.array(self.get_non_attackers_neighbors(self.G, self.attackers))
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

            B_t = W_TT @ B_t+ W_TA @ X_A_t.to(torch.float64)

        v = torch.stack(new_v).numpy()
        
        try:
            return np.linalg.lstsq(self.target_knowledge_matrix, v)[0]
        except Exception as e:
            print(e)
            print("Building the knowledge matrix failed")
            raise


# based on InvertingGradients code by Jonas Geiping
# code found in https://github.com/JonasGeiping/invertinggradients/tree/1157b61c6704df42c497ab9eb074c75da5204334

def compute_param_delta(param_s, param_t, basic_params):
    """
    Generates the input value for reconstruction
    Assumes param_s and param_t are from the same client.

    basic_params: list of names present in model params
    """
    assert len(param_s) != 0 and len(param_t) != 0, "Empty parameters"
    return [(param_t[name].to("cuda") - param_s[name].to("cuda")).detach() for name in basic_params if name in param_s and name in param_t]

def reconstruct_gradient(param_diff, 
                         target_labels, 
                         target_images, 
                         lr, 
                         local_steps, 
                         model, 
                         mean,
                         std,
                         client_id=0, 
                         gradient_reconstructor=False,
                         dataset="cifar10") -> Tuple[torch.Tensor, float, float, float]:
    """
    Reconstructs the gradient following the Geiping InvertingGradients technique
    """

    if type(mean) == type(std) == float:
        mean, std = (mean,), (std,)
    setup = inversefed.utils.system_startup()
    for p in range(len(param_diff)):
        param_diff[p] = param_diff[p].to(setup['device'])
    # param_diff = param_diff.to(setup['device'])
    target_labels = target_labels.to(setup['device'])
    target_images = target_images.to(setup['device'])

    dm = torch.as_tensor(mean, **setup)[:, None, None]
    ds = torch.as_tensor(std, **setup)[:, None, None]
    img_shape = (3,32,32) if dataset == "cifar10" else (1,28,28)
    
    model = model.to(setup['device'])

    config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                # lr=1,
                lr=0.1,
                optim='adam',
                restarts=1,
                # max_iterations=8_000,
                max_iterations=2_000,
                # total_variation=1e-4,
                total_variation=1e-1,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    
    # assert len(param_diff) == 38 # hard coded for resnet18

    if gradient_reconstructor:
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
    else:
        rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, lr, config,
                                                    use_updates=True, num_images=len(target_labels), batch_size=1)
    

    result = rec_machine.reconstruct(param_diff, target_labels, img_shape=img_shape)
    
    # Check for NaN in the reconstruction loss
    if result == None:
        print(f"Reconstruction failed for client {client_id} due to NaN in the reconstruction loss.")
        return None

    output, stats = result
    # compute reconstruction acccuracy
    test_mse = (output.detach() - target_images).pow(2).mean()
    feat_mse = (model(output.detach())- model(target_images)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, target_images, factor=1/ds)
    # Convert test_mse and feat_mse to floats
    test_mse = test_mse.item()
    feat_mse = feat_mse.item()
    return output, test_mse, test_psnr, feat_mse

    print(f"Client {client_id} Test MSE: {test_mse:.2e}, Test PSNR: {test_psnr:.2f}, Feature MSE: {feat_mse:.2e}")

    grid_plot(output, target_labels, ds, dm, stats, test_mse, feat_mse, test_psnr, save_path=f"gias_output_client_{client_id}.png")    
    return output, test_mse, test_psnr, feat_mse

def grid_plot(tensor, labels, ds, dm, stats, test_mse, feat_mse, test_psnr, save_path=None):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)

    fig, axes = plt.subplots(1, 10, figsize=(24, 24))
    for im, l, ax in zip(tensor, labels, axes.flatten()):
        ax.imshow(im.permute(1, 2, 0).cpu())
        ax.set_title(l)
        ax.axis('off')
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
            f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure if save_path is provided
    # plt.show()  # Show the plot after saving

def gia_main(param_s, 
             param_t, 
             base_params, 
             model, 
             target_labels, 
             target_images, 
             client_id, 
             device, 
             mean,
             std,
             lr=3e-4,) -> None | Tuple[torch.Tensor, float, float, float]:
    """
    Main function for Gradient Inversion Attack
    Returns results moved back to their original devices
    """
    
    # Store original parameter devices
    param_s_devices = {name: param_s[name].device for name in base_params if name in param_s}
    param_t_devices = {name: param_t[name].device for name in base_params if name in param_t}
    
    param_diff = compute_param_delta(param_s, param_t, base_params)

    # Check if all elements in para_diff are zero tensors
    if all((diff == 0).all() for diff in param_diff):
        print("Parameter differences contain only zeros for client ", client_id)
        return None  # or return an empty list, depending on your needs
    
    output, test_mse, test_psnr, feat_mse = reconstruct_gradient(param_diff=param_diff, 
                                                                 target_labels=target_labels, 
                                                                 target_images=target_images, 
                                                                 lr=lr,
                                                                 local_steps=5, 
                                                                 model=model, 
                                                                 client_id=client_id,
                                                                 mean=mean,
                                                                 std=std)
                
    # Move output back to target_images device (since it's a reconstruction of the images)
    if output is not None:
        output = output.to(device)
    
    # Move model back to original device
    model = model.to(device)
    
    # Move parameters back to their original devices
    for name in base_params:
        if name in param_s:
            param_s[name] = param_s[name].to(param_s_devices[name])
        if name in param_t:
            param_t[name] = param_t[name].to(param_t_devices[name])
            
    # Move labels and images back to their original devices
    target_labels = target_labels.to(device)
    target_images = target_images.to(device)
            
    return output, test_mse, test_psnr, feat_mse

def deflatten(t, base_params):
    """
    Function to recover the original parameter configuration from the flattened tensor t

    from https://github.com/AbdellahElmrini/decAttack/blob/master/src/models.py
    """
    param = list(base_params)
    n_params = len(param)
    assert len(torch.cat([param[j].flatten() for j in range(n_params)])) == len(t), f"Length of flattened tensor {len(t)} does not match the number of parameters {len(torch.cat([param[j].flatten() for j in range(n_params)]))}"
    res = []
    i = 0
    for j in range(n_params):
        d = len(param[j].flatten())
        res.append(t[i:i+d].reshape(param[j].shape))
        i+=d
    return res

def gia_from_disambiguate(x_hat_i, base_params, model, target_labels, target_images, client_id, lr, device, mean, std, dataset) -> None | Tuple[torch.Tensor, float, float, float]:
    """
    Function to perform Gradient Inversion Attack from disambiguated gradients
    """
    
    # assert model_device == target_images_device == target_labels_device, f"Model and data devices do not match. Model device: {model_device}, Target images device: {target_images_device}, Target labels device: {target_labels_device}"
    # Move x_hat to the device of the model
    setup = inversefed.utils.system_startup()
    x_hat = x_hat_i.to(device)
    input_gradient = deflatten(-1/lr*x_hat_i, base_params)
    input_gradient = [torch.tensor(el , **setup) for el in input_gradient]

    print(f"reconstructing gradient for client {client_id}")

    out = reconstruct_gradient(param_diff=input_gradient, 
                                                                 target_labels=target_labels, 
                                                                 target_images=target_images, 
                                                                 lr=lr, 
                                                                 local_steps=1, 
                                                                 model=model, 
                                                                 mean=mean,
                                                                 std=std,
                                                                 client_id=client_id, 
                                                                 gradient_reconstructor=True,
                                                                 dataset=dataset)
    
    # Move model back to original device
    model = model.to(device)
    
    # Move labels and images back to their original devices
    target_labels = target_labels.to(device)
    target_images = target_images.to(device)

    if out is not None:
        output, test_mse, test_psnr, feat_mse = out
        if dataset == "mnist":
            output = output.reshape(1,1,28,28)
        elif dataset == "cifar10":
            output = output.reshape(1,3,32,32)
        return output, test_mse, test_psnr, feat_mse
    
    return None