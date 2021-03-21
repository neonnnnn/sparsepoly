import numpy as np
from numba import float64, boolean
from numba.experimental import jitclass
from .utils import norm
from math import sqrt

spec = [
    ("_norms", float64[:]),
    ("_cache", float64[:]),
    ("_dcache", float64[:]),
    ("_cache_all_subsets", float64)
]


@jitclass(spec)
class OmegaCS(object):
    def __init__(self):
        pass

    def _eval(self, Ps, degree):
        n = len(Ps)
        n_features = Ps.shape[1]
        cache = np.zeros((degree+1, n))
        cache[0] = 1.0
        norms = np.asfortranarray(norm(Ps, ord=2, axis=-1))
        for j in range(n_features):
            for deg in range(degree):
                cache[degree-deg] += cache[degree-deg-1] * norms[:, j]
        return cache[-1]

    def __call__(self, P, degree):
        shape = P.shape[:-2]
        result = self._eval(P.reshape(-1, P.shape[-1], P.shape[-2]), degree)
        if P.ndim == 2:
            return result[0]
        else:
            return result.reshape(shape)

    def init_cache_pbcd(self, degree, n_features, n_components):
        self._norms = np.zeros(n_features)
        if degree > 0:
            self._cache = np.zeros(degree+1)
            self._dcache = np.zeros(degree+1)
            self._dcache[1] = 1.0
        elif degree == -1:
            self._cache_all_subsets = 1.0
        else:
            raise ValueError("degree must be a positive int or -1.")

    def __recompute_cache_bcd(self, degree):
        if degree > 0: # factorization machine
            self._cache[1:] = 0.0
            self._cache[0] = 1.0
            for norm in self._norms:
                for deg in range(degree):
                    self._cache[degree-deg] += self._cache[degree-deg-1] * norm
        else: # all-subsets
            self._cache_all_subsets = 1.0
            for l2 in self._norms:
                self._cache_all_subsets *= (1.0 + l2)

    def compute_cache_pbcd(self, P, degree):
        self._norms[:] = norm(P, ord=2, axis=1)
        self.__recompute_cache_bcd(degree)
        
    def update_cache_pbcd(self, P, degree, j):
        l2 = sqrt(np.dot(P[j], P[j]))
        if degree > 0: # factorization machine
            for deg in range(1, degree+1):
                self._cache[deg] += self._dcache[deg] * l2
                self._cache[deg] -= self._dcache[deg] * self._norms[j]
            self._norms[j] = l2
            if min(self._cache) < 0:
                self.__recompute_cache_bcd(degree)
        else: # all-subsets
            self._cache_all_subsets *= 1.0 + l2
            self._norms[j] = l2
            if self._cache_all_subsets < 0:
                self.__recompute_cache_bcd(-1)

    def prox_bcd(self, p_j, strength, degree, j):
        l2 = sqrt(np.dot(p_j, p_j))
        if degree > 0: # factorization machine
            for deg in range(2, degree+1):
                self._dcache[deg] = self._cache[deg-1]
                self._dcache[deg] -= self._dcache[deg-1] * self._norms[j]

            if min(self._dcache) < 0: # numerical error
                self._norms[j] = 0.0
                self.__recompute_cache_bcd(degree-1)
                self._dcache[0] = 0.0
                self._dcache[1] = 1.0
                for deg in range(2, degree+1):
                    self._dcache[deg] = self._cache[degree-1]
            
            strength *= self._dcache[degree]
        else: # all-subsets
            self._cache_all_subsets /= 1.0 + self._norms[j]
            strength *= self._cache_all_subsets

        if l2 > strength:
            p_j *= (1 - strength / l2)
        else:
            p_j[:] = 0.0
