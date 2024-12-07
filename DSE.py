import numpy as np
from sklearn.metrics import pairwise_distances
import warnings
import random

warnings.filterwarnings("ignore")

"""
    Adapted from Liao et al. 2023
    Assessing Neural Network Representations During Training Using Noise-Resilient Diffusion Spectral Entropy
"""


def compute_diffusion_matrix(X: np.array, sigma: float = 10.0):
    '''
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X returns a diffusion matrix P, as an numpy ndarray.
    Using the "anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        sigma: a float
            conceptually, the neighborhood size of Gaussian kernel.
    Returns:
        K: a numpy array of size n x n that has the same eigenvalues as the diffusion matrix.
    '''

    # Construct the distance matrix.
    D = pairwise_distances(X)

    # Gaussian kernel
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))

    # Anisotropic density normalization.
    Deg = np.diag(1 / np.sum(G, axis=1)**0.5)
    K = Deg @ G @ Deg

    # Now K has the exact same eigenvalues as the diffusion matrix `P`
    # which is defined as `P = D^{-1} K`, with `D = np.diag(np.sum(K, axis=1))`.

    return K

import numpy as np

def exact_eigvals(A: np.array):
    '''
    Compute the exact eigenvalues.
    '''
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix.
        eigenvalues = np.linalg.eigvalsh(A)
    else:
        eigenvalues = np.linalg.eigvals(A)

    return eigenvalues


def exact_eig(A: np.array):
    '''
    Compute the exact eigenvalues & vecs.
    '''

    #return np.ones(A.shape[0]), np.ones((A.shape[0],A.shape[0]))
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix.
        eigenvalues_P, eigenvectors_P = np.linalg.eigh(A)
    else:
        eigenvalues_P, eigenvectors_P = np.linalg.eig(A)

    # Sort eigenvalues
    sorted_idx = np.argsort(eigenvalues_P)[::-1]
    eigenvalues_P = eigenvalues_P[sorted_idx]
    eigenvectors_P = eigenvectors_P[:, sorted_idx]
    return eigenvalues_P, eigenvectors_P


def diffusion_spectral_entropy(embedding_vectors: np.array,
                               gaussian_kernel_sigma: float = 10,
                               t: int = 1,
                               random_seed: int = 29):
    '''

    Diffusion Spectral Entropy over a set of N vectors, each of D dimensions.

    DSE = - sum_i [eig_i^t log eig_i^t]
        where each `eig_i` is an eigenvalue of `P`,
        where `P` is the diffusion matrix computed on the data graph of the [N, D] vectors.


    args:
        embedding_vectors: np.array of shape [N, D]
            N: number of data points / samples
            D: number of feature dimensions of the neural representation

        gaussian_kernel_sigma: float
            The bandwidth of Gaussian kernel (for computation of the diffusion matrix)
            Can be adjusted per the dataset.
            Increase if the data points are very far away from each other.

        t: int
            Power of diffusion matrix (equivalent to power of diffusion eigenvalues)
            <-> Iteration of diffusion process
            Usually small, e.g., 1 or 2.
            Can be adjusted per dataset.
            Rule of thumb: after powering eigenvalues to `t`, there should be approximately
                           1 percent of eigenvalues that remain larger than 0.01

    '''

    random.seed(random_seed)


    K = compute_diffusion_matrix(embedding_vectors,sigma=gaussian_kernel_sigma)
    eigvals = exact_eigvals(K)

    # Eigenvalues may be negative. Only care about the magnitude, not the sign.
    eigvals = np.abs(eigvals)

    # Power eigenvalues to `t` to mitigate effect of noise.
    eigvals = eigvals**t

    prob = eigvals / eigvals.sum()

    prob = prob + np.finfo(float).eps
    entropy = -np.sum(prob * np.log2(prob))

    return entropy
