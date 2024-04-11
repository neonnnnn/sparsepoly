from math import sqrt

import numpy as np
from numba import boolean, float64
from numba.experimental import jitclass

from .utils import norm

spec = [
    ("transpose", boolean),
]


@jitclass(spec)
class L21(object):
    def __init__(self, transpose=False):
        self.transpose = transpose

    def eval(self, P):
        axis = -2 if self.transpose else -1
        return norm(norm(P, ord=2, axis=axis), ord=1, axis=-1)

    def init_cache_pbcd(self, degree, n_features, n_components):
        if self.transpose:
            raise ValueError("self.transpose is True.")

    def compute_cache_pbcd(self, P, degree):
        pass

    def update_cache_pbcd(self, P, degree, j):
        pass

    def prox_bcd(self, p_j, strength, degree, j):
        l2 = sqrt(np.dot(p_j, p_j))
        if l2 > strength:
            p_j *= 1.0 - strength / l2
        else:
            p_j[:] = 0.0

    def init_cache_psgd(self, degree, n_features, n_components):
        pass

    def prox(self, P, strength, degree):
        # assume P.shape = (n_features, n_components)
        axis = 0 if self.transpose else 1
        norms = norm(P, 2, axis=axis)
        norms[norms <= strength] = np.inf
        P *= 1.0 - strength / np.expand_dims(norms, axis=axis)
