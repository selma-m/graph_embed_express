import torch
import numpy as np
import networkx as nx
from numpy import linalg as LA
import scipy.stats.mstats
from itertools import product 

"""
    Implementation of vanilla geometric scattering and BLIS-inspired scattering
"""

torch.manual_seed(29)
np.random.seed(29)

def lazy_random_walk(A):
    # input is adjacency matrix
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    d = A.sum(0) # sum along columns
    P_t = A/d 
    P_t[torch.isnan(P_t)] = 0
    identity_matrix = torch.eye(P_t.shape[0], dtype=P_t.dtype)
    P = 0.5 * (identity_matrix + P_t)
    return P

def relu(x):
    return x * (x > 0)

def reverse_relu(x):
    return relu(-x)

def graph_wavelet(P):
    psi = []
    for d1 in [1,2,4,8,16]: # these are the scales
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        W_d1_tensor = torch.tensor(W_d1, dtype=torch.float32)
        psi.append(W_d1_tensor)
    psi.append(torch.tensor(LA.matrix_power(P,2*16), dtype=torch.float32))
    return psi

def graph_wavelet1(A, largest_scale = 16): 
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    d_arr = A.sum(1) 
    d_arr_inv = 1/d_arr 
    d_arr_inv[torch.isnan(d_arr_inv)] = 0 
    D_inv_sqrt = torch.diag(np.sqrt(d_arr_inv))
    T = 0.5 * (torch.eye(A.shape[0]) + D_inv_sqrt @ A @ D_inv_sqrt)
    w, U = LA.eigh(T) #return eigenvalues and eigenvectors
    w = torch.tensor(w, dtype = torch.float32)
    w = torch.maximum(w, torch.tensor(0.0)) # ReLU operation

    M_inv = torch.diag(torch.sqrt(d_arr))
    wavelets = torch.zeros((largest_scale + 2, *T.shape))
    eig_filter = torch.sqrt(torch.maximum(torch.ones(len(w)) - w, torch.tensor(0.0)))
    Psi = M_inv @ U @ torch.diag(eig_filter) @ U.T @ D_inv_sqrt 
    wavelets[0,:,:] = Psi

    for scale in range(1, largest_scale + 1):
        eig_filter = torch.sqrt(torch.maximum(w ** (2 **(scale-1) ) - w ** (2 ** scale), torch.tensor(0.0)))
        Psi = M_inv @ U @ torch.diag(eig_filter) @ U.T @ D_inv_sqrt
        wavelets[scale,:,:] = Psi

    low_pass = M_inv @ U @ torch.diag(torch.sqrt(w ** (2 ** largest_scale))) @ U.T @ D_inv_sqrt
    wavelets[-1,:,:] = low_pass

    return wavelets



def node_level(x,A):
    P = lazy_random_walk(A)
    psi = graph_wavelet(P)

    new_x = x
    for i in range(len(psi)):
        new_x = psi[i]@new_x # multiply with wavelet
        if i < len(psi): # don't apply linearity the last time
            new_x = np.abs(new_x) # non-linearity

    return new_x

def graph_level(x,A):
    node_level_signal = node_level(x,A)
    graph_level_signal = torch.abs(node_level_signal).sum(0) # simply sum over nodes (rows)
    return graph_level_signal

def node_level_blis(x,A, num_layers = 10):
    P = lazy_random_walk(A)
    psi = graph_wavelet(P)
    new_x = x
    for i in range(len(psi)):
        new_x = psi[i]@new_x # multiply with wavelet
        new_x = np.maximum(new_x, 0) + np.maximum(-new_x,0)# non-linearity

    return new_x


def graph_level_blis(x,A):
    node_level_signal = node_level_blis(x,A)
    graph_level_signal = node_level_signal.sum(0)
    return graph_level_signal
