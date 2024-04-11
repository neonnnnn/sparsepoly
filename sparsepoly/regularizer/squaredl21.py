import numpy as np
from math import sqrt
from numba import float64, boolean, int32
from numba.experimental import jitclass
from .utils import norm, prox_squaredl12

spec = [
    ("transpose", boolean),
    ("strength", float64),
    ("_norms", float64[:]),
    ("_cache", float64),
    ("_candidates", int32[:])
]

@jitclass(spec)
class SquaredL21(object):
    def __init__(self, transpose=False):
        self.transpose = transpose

    def eval(self, P):
        axis = -2 if self.transpose else -1
        return norm(norm(P, ord=2, axis=axis), ord=1, axis=-1) ** 2
    
    def init_cache_pbcd(self, degree, n_features, n_components):
        if degree != 2:
            raise ValueError("SquaredL21 supports only degree=2.")

        if self.transpose:
            raise ValueError("transpose != False.")
        self._norms = np.zeros(n_features)
        self._cache = 0
    
    def compute_cache_pbcd(self, P, degree):
        self._norms[:] = norm(P, 2, axis=1)
        self._cache = np.sum(self._norms)

    def update_cache_pbcd(self, P, degree, j):
        self._cache -= self._norms[j]
        self._norms[j] = sqrt(np.dot(P[j], P[j]))
        self._cache += self._norms[j]
    
    def prox_bcd(self, p_j, strength, degree, j):
        p_j /= (1+2*strength)
        l2 = np.sqrt(np.dot(p_j, p_j))
        if self._cache < self._norms[j]: # to avoid numerical error
            self._cache = np.sum(self._norms)
        dcache = self._cache - self._norms[j]
        strength = 2 * dcache * strength / (1.0 + 2 * strength)
        if l2 > strength:
            p_j *= (1.0 - strength / l2)
        else:
            p_j[:] = 0.0

    def init_cache_psgd(self, degree, n_features, n_components):
        if self.transpose:
            self._candidates = np.arange(n_components, dtype=np.int32)
        else:
            self._candidates = np.arange(n_features, dtype=np.int32)

    def prox(self, P, strength, degree):
        # assume P.shape = (n_features, n_components)
        axis = 0 if self.transpose else 1
        norms = norm(P, 2, axis)
        idx = norms > 0
        if self.transpose:
            P[:, idx] /= norms[idx]
        else:
            P[idx] /= norms[idx].reshape(-1, 1)
        prox_squaredl12(norms, strength, self._candidates)
        norms = np.expand_dims(norms, axis=axis)
        P *= norms
