from .model import HMM
from . import toolkit as tk
import numpy as np


def build_Gaussian_HOHMM(K: int, r: int, m: int, dim: int, init_mean=0, init_sacle=0.1):

    nu = max(r, m)
    # init mu and Sigma
    phi = tk.init_gaussian_emission(K, m, dim, init_mean, init_sacle)
    # init transition matrix
    A = tk.init_transit_matrix(K, r, m)
    # init initial distribution
    pi = tk.init_hidden_state_dist(np.power(K, nu))

    obj = HMM(A=A, phi=phi, pi=pi, num_states=K, r=r, m=m, max_perd=2000)
    return obj


if __name__ == '__main__':
    print('test')