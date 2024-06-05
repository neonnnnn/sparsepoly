import numpy as np
from numba import boolean
from numba.experimental import jitclass

from sparsepoly.regularizer.utils import norm, soft_thresholding

spec = [
    ("transpose", boolean),
]


@jitclass(spec)
class L1(object):
    def __init__(self):
        pass

    def eval(self, P, degree):
        return norm(P, ord=1, axis=(-2, -1))

    def init_cache_pcd(self, degree, n_features, n_components):
        pass

    def compute_cache_pcd_all(self, P, degree):
        pass

    def compute_cache_pcd(self, P, degree, s):
        pass

    def update_cache_pcd(self, P, degree, s, j):
        pass

    def prox_cd(self, p_sj, strength, degree, j):
        return np.sign(p_sj) * max(abs(p_sj) - strength, 0.0)

    def init_cache_pbcd(self, degree, n_features, n_components):
        pass

    def compute_cache_pbcd(self, P, degree):
        pass

    def update_cache_pbcd(self, P, degree, j):
        pass

    def prox_bcd(self, p_j, strength, degree, j):
        soft_thresholding(p_j, strength)

    def init_cache_psgd(self, degree, n_features, n_components):
        pass

    def prox(self, P, strength, degree):
        soft_thresholding(P, strength)
