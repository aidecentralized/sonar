# Taken from https://github.com/yukiTakezawa/BaseGraph

# Part of @inproceedings{takezawa2023exponential,
#      title={Beyond Exponential Graph: Communication-Efficient Topologies for Decentralized Learning via Finite-time Convergence}, 
#      author={Yuki Takezawa and Ryoma Sato and Han Bao and Kenta Niwa and Makoto Yamada},
#      year={2023},
#      booktitle={NeurIPS}
#}

import torch
import math
import copy
import sympy
import numpy as np
import networkx as nx
import math

class DynamicGraph():
    def __init__(self, w_list):
        """
        Parameter
        --------
        w_list (list of torch.tensor):
            list of mixing matrix
        """
        self.w_list = w_list
        self.n_nodes = w_list[0].size()[0]
        self.length = len(w_list)
        self.itr = 0
        
    def get_in_neighbors(self, i):
        """
        Parameter
        ----------
        i (int):
            a node index
        Return
        ----------
            dictionary of (neighbors's index: weight of the edge (i,j))
        """
        w = self.w_list[self.itr%self.length]        

        return {idx.item(): w[idx, i].item() for idx in torch.nonzero(w[:,i])}

    def get_out_neighbors(self, i):
        """
        Parameter
        ----------
        i (int):
            a node index
        Return
        ----------
            dictionary of (neighbors's index: weight of the edge (i,j))
        """
        w = self.w_list[self.itr%self.length]        
        
        return {idx.item(): w[i,idx].item() for idx in torch.nonzero(w[i])}

    
    def get_neighbors(self, i):
        in_neighbors = self.get_in_neighbors(i)
        out_neighbors = self.get_out_neighbors(i)
        self.itr += 1
        return in_neighbors, out_neighbors
        
    
    def get_w(self):
        w = self.w_list[self.itr%self.length]        
        self.itr += 1
        return w


class OnePeerExponentialGraph(DynamicGraph):
    def __init__(self, n_nodes):
        w_list = []

        n_neighbors = int(math.log2(n_nodes-1))

        for j in range(n_neighbors+1):
            
            w = torch.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                w[i,i] = 1/2
                w[i, (i+2**j)%n_nodes] = 1/2
                
            w_list.append(w)
            
        super().__init__(w_list)


class HyperHyperCube(DynamicGraph):
    def __init__(self, n_nodes, seed=0, max_degree=1):
        self.state = np.random.RandomState(seed)
        self.max_degree = max_degree
        
        if n_nodes == 1:
            super().__init__([torch.eye(1)])
        else:
            if list(sympy.factorint(n_nodes))[-1] > max_degree+1:
                print(f"Can not construct {max_degree}-peer graphs")
        
            node_list = list(range(n_nodes))
            factors_list = self.split_node(node_list, n_nodes)
            #print(factors_list)
            super().__init__(self.construct(node_list, factors_list, n_nodes))
    
    def construct(self, node_list, factors_list, n_nodes):
        w_list = []
        for k in range(len(factors_list)):
            #print(factors_list)
            
            w = torch.zeros((n_nodes, n_nodes))
            b = torch.zeros(n_nodes)
            
            for i_idx in range(len(node_list)):
                for nk in range(1, factors_list[k]):
                    
                    i = node_list[i_idx]
                    j = int(i + np.prod(factors_list[:k]) * nk) % n_nodes
                    
                    if b[i] < factors_list[k]-1 and b[j] < factors_list[k]-1:
                        #print("g", i, j, b[i], b[j])
                        b[i] += 1
                        b[j] += 1
                        
                        w[i, j], w[j, i] = 1/factors_list[k], 1/factors_list[k]
                        w[i, i], w[j, j] = 1/factors_list[k], 1/factors_list[k]

            w_list.append(w)
       
        return w_list
                    

    def split_node(self, node_list, n_nodes):
        factors_list = []
        rest = n_nodes
        
        for factor in reversed(range(2, self.max_degree+2)):
            while rest % factor == 0:
                factors_list.append(factor)
                rest = int(rest / factor)

                if rest == 1:
                    break

        factors_list.reverse()
        return factors_list

    
class SimpleBaseGraph(DynamicGraph):
    def __init__(self, n_nodes, max_degree=1, seed=0, inner_edges=True):
        self.state = np.random.RandomState(seed)
        self.inner_edges = inner_edges
        self.max_degree = max_degree
        self.n_nodes = n_nodes

        super().__init__(self.construct())

    def construct(self):
        node_list_list, n_nodes_list = self.split_nodes()
        node_list_list_list = self.split_nodes2(node_list_list)
        L = len(node_list_list)
        
        if self.n_nodes == 1:
            return [torch.eye(1)]
        elif max(list(sympy.factorint(self.n_nodes))) <= self.max_degree + 1:
            return HyperHyperCube(self.n_nodes, max_degree=self.max_degree).w_list
        
        # construct k-peer HyperHyperCube
        hyperhyper_cubes = [HyperHyperCube(len(node_list_list[i]), max_degree=self.max_degree) for i in range(L)]        
        hyperhyper_cubes2 = [HyperHyperCube(len(node_list_list_list[i][0]), max_degree=self.max_degree) for i in range(L)]
        max_length_of_hyper = len(hyperhyper_cubes[0].w_list)

        b = torch.zeros(L)
        true_b = torch.tensor([len(hyperhyper_cube.w_list) for hyperhyper_cube in hyperhyper_cubes2])
        
        w_list = []
        m = -1
        while True:
            m += 1
            w = torch.zeros((self.n_nodes, self.n_nodes))
            isolated_nodes = None
            all_isolated_nodes = None
            
            for l in reversed(range(L)):
                
                if m < max_length_of_hyper:
                    length = len(hyperhyper_cubes[l].w_list)
                    w += self.extend(hyperhyper_cubes[l].w_list[m % length], node_list_list[l])
                    
                elif m < max_length_of_hyper + l:
                    if isolated_nodes is None:
                        isolated_nodes = copy.deepcopy(node_list_list_list[m - max_length_of_hyper])
                        all_isolated_nodes = [node for nodes in isolated_nodes for node in nodes]
                        
                    for i in node_list_list[l]:
                        a_l = len(isolated_nodes)
                        
                        for k in range(a_l):
                            j = isolated_nodes[k].pop(-1)
                            all_isolated_nodes.remove(j)
                            w[i, j] = n_nodes_list[m - max_length_of_hyper] / sum(n_nodes_list[m - max_length_of_hyper:]) / a_l
                            w[j, i] = n_nodes_list[m - max_length_of_hyper] / sum(n_nodes_list[m - max_length_of_hyper:]) / a_l

                            w[j, j] = 1 - w[i, j]
                        w[i, i] = 1 - n_nodes_list[m - max_length_of_hyper] / sum(n_nodes_list[m - max_length_of_hyper:])
                            
                elif m == max_length_of_hyper + l and l != L-1:
                    while len(all_isolated_nodes) > 1 and self.inner_edges:
                        sampled_nodes = all_isolated_nodes[:min(self.max_degree+1,len(all_isolated_nodes))]

                        for node_id in sampled_nodes:
                            all_isolated_nodes.remove(node_id)
                        
                        for i in sampled_nodes:
                            for j in sampled_nodes:
                                w[i, j] = 1 / len(sampled_nodes)
                                w[j, i] = 1 / len(sampled_nodes)
                                w[i, i] = 1 / len(sampled_nodes)
                                w[j, j] = 1 / len(sampled_nodes) 
            
                else:
                    if n_nodes_list[l] < self.max_degree+1:
                        length = len(hyperhyper_cubes[l].w_list)
                        w += self.extend(hyperhyper_cubes[l].w_list[int(b[l] % length)], node_list_list[l])
                    else:
                        a_l = len(node_list_list_list[l])
                        
                        for k in range(a_l):
                            length = len(hyperhyper_cubes2[l].w_list)
                            w += self.extend(hyperhyper_cubes2[l].w_list[int(b[l] % length)], node_list_list_list[l][k])
                        
                    b[l] += 1

            # add self-loop
            for i in range(self.n_nodes):
                if w[i, i] == 0:
                    w[i,i] = 1.0
            w_list.append(w)

            #if (b >= true_b).all():
            #    break
            if b[0] == len(hyperhyper_cubes2[0].w_list):
                break
            
        return w_list
            
    def diag(self, X, Y):
        new_W = torch.zeros((X.size()[0] + Y.size()[0], X.size()[0] + Y.size()[0]))
        new_W[0:X.size()[0], 0:X.size()[0]] = X
        new_W[X.size()[0]:, X.size()[0]:] = Y
        return new_W


    def extend(self, w, node_list):
        new_w = torch.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w

    def split_nodes(self):
        factor = (self.max_degree + 1)**int(math.log(self.n_nodes, self.max_degree+1))
        n_nodes_list = []
        
        while sum(n_nodes_list) != self.n_nodes:

            rest = self.n_nodes - sum(n_nodes_list)
            
            if rest >= factor:
                n_nodes_list.append((rest // factor) * factor)
            factor = int(factor/(self.max_degree  + 1))
        node_list = list(range(self.n_nodes))
        node_list_list = []
        for i in range(len(n_nodes_list)):
            node_list_list.append(node_list[sum(n_nodes_list[:i]):sum(n_nodes_list[:i+1])])

        return node_list_list, n_nodes_list

    
    def split_nodes2(self, node_list_list):
        """
        len(node_list) can be written as a_l * (max_degree + 1)^{p_l} where al \in \{1, 2, \cdots, k\}.
        """

        node_list_list_list = []
        
        for node_list in node_list_list:
            n_nodes = len(node_list)
            power = math.gcd(n_nodes, (self.max_degree+1) ** int(math.log(n_nodes, self.max_degree+1)))
            rest = int(n_nodes / power)

            node_list_list_list.append([])
            for i in range(rest):
                node_list_list_list[-1].append(node_list[i*power:(i+1)*power])
                
        return node_list_list_list
    

class BaseGraph(DynamicGraph):
    def __init__(self, n_nodes, max_degree=1, seed=0, inner_edges=True):
        self.state = np.random.RandomState(seed)
        self.inner_edges = inner_edges
        self.max_degree = max_degree
        self.n_nodes = n_nodes
        self.seed = seed
        
        super().__init__(self.construct())

    def construct(self):
        node_list_list1, node_list_list2, n_power, n_rest = self.split_nodes()

        simple_adics = [SimpleBaseGraph(len(node_list_list1[i]), max_degree=self.max_degree) for i in range(n_power)]
        hyper_cubes = [HyperHyperCube(len(node_list_list2[i]), max_degree=self.max_degree) for i in range(n_rest)]

        # check which is better
        g = SimpleBaseGraph(self.n_nodes, max_degree=self.max_degree, seed=self.seed, inner_edges=self.inner_edges)
        if len(g.w_list) < len(simple_adics[0].w_list) + len(hyper_cubes[0].w_list):
            return g.w_list
        
        
        w_list = []
        for m in range(len(simple_adics[0].w_list)):
            w = torch.zeros((self.n_nodes, self.n_nodes))
            
            for l in range(n_power):
                w += self.extend(simple_adics[l].w_list[m], node_list_list1[l])
            w_list.append(w)
            
        for m in range(len(hyper_cubes[0].w_list)):
            w = torch.zeros((self.n_nodes, self.n_nodes))
            
            for l in range(n_rest):
                w += self.extend(hyper_cubes[l].w_list[m], node_list_list2[l])
            w_list.append(w)

        return w_list
    
        
    def diag(self, X, Y):
        new_W = torch.zeros((X.size()[0] + Y.size()[0], X.size()[0] + Y.size()[0]))
        new_W[0:X.size()[0], 0:X.size()[0]] = X
        new_W[X.size()[0]:, X.size()[0]:] = Y
        return new_W


    def extend(self, w, node_list):
        new_w = torch.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w
    
    
    def split_nodes(self):
        factors = [n**int(math.log(self.n_nodes, n)) for n in range(2, self.max_degree+2)]
        factor = np.prod(factors) 
        n_power = math.gcd(self.n_nodes, factor)
        n_rest = int(self.n_nodes / n_power)

        node_list = list(range(self.n_nodes))
        node_list_list1 = []
        for i in range(n_power):
            node_list_list1.append(node_list[n_rest*i:n_rest*(i+1)])

        node_list_list2 = [[] for _ in range(n_rest)]
        for i in range(n_power):
            for j in range(n_rest):
                node_list_list2[j].append(node_list_list1[i][j])
            
        return node_list_list1, node_list_list2, n_power, n_rest

    
    def get_neighbors(self, i):
        in_neighbors = self.get_in_neighbors(i)
        out_neighbors = self.get_out_neighbors(i)
        self.itr += 1

        #if self.itr % len(self.w_list) == 0:
        #    self.w_list = self.shuffle_node_index(self.w_list, self.n_nodes)

        return in_neighbors, out_neighbors        