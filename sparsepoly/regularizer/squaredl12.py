import numpy as np
from numba import boolean, float64, int32
from numba.experimental import jitclass

from .utils import norm, prox_squaredl12

spec = [
    ("transpose", boolean),
    ("_cache", float64[:]),
    ("_abs_p", float64[:]),
    ("_candidates", int32[:]),
]


@jitclass(spec)
class SquaredL12(object):
    def __init__(self, transpose=True):
        self.transpose = transpose

    def eval(self, P):
        axis = -2 if self.transpose else -1
        return norm(norm(P, ord=1, axis=axis), axis=-1) ** 2

    def init_cache_pcd(self, degree, n_features, n_components):
        if degree > 2:
            raise ValueError("SquaredL12 supports only degree=2.")
        self._abs_p = np.zeros(n_features)
        if self.transpose:
            self._cache = np.zeros(1)
        else:
            self._cache = np.zeros(n_features)

    def compute_cache_pcd_all(self, P, degree):
        if not self.transpose:
            n_components = P.shape[0]
            n_features = P.shape[1]
            for j in range(n_features):
                self._cache[j] = 0.0
                for s in range(n_components):
                    self._cache[j] += abs(P[j, s])

    def compute_cache_pcd(self, P, degree, s):
        self._abs_p[:] = np.abs(P[s])
        if self.transpose:
            self._cache[0] = np.sum(self._abs_p)

    def update_cache_pcd(self, P, degree, s, j):
        i = 0 if self.transpose else j
        self._cache[i] -= self._abs_p[j]
        self._cache[i] += np.abs(P[s, j])

    def prox_cd(self, p_sj, strength, degree, j):
        i = 0 if self.transpose else j
        dcache = self._cache[i] - self._abs_p[j]
        p_sj /= 1 + 2 * strength
        sign = 1 if p_sj > 0 else -1
        return sign * max(abs(p_sj) - 2 * strength * dcache / (1 + 2 * strength), 0)

    def init_cache_psgd(self, degree, n_features, n_components):
        if self.transpose:
            self._cache = np.zeros(n_features)
            self._candidates = np.arange(n_features, dtype=np.int32)
        else:
            self._candidates = np.arange(n_components, dtype=np.int32)

    def prox(self, P, strength, degree):
        # assume P.shape = (n_features, n_components)
        n_features, n_components = P.shape
        if self.transpose:
            for s in range(n_components):
                for j in range(n_features):
                    self._cache[j] = P[j, s]
                prox_squaredl12(self._cache, strength, self._candidates)
                for j in range(n_features):
                    P[j, s] = self._cache[j]
        else:
            for j in range(n_features):
                prox_squaredl12(P[j], strength, self._candidates)
