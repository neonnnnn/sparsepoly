import numpy as np
from numba import boolean, float64
from numba.experimental import jitclass

spec = [
    ("transpose", boolean),
    ("_cache", float64[:]),
    ("_abs_p", float64[:]),
    ("_dcache", float64[:]),
    ("_cache_all_subsets", float64),
]


@jitclass(spec)
class OmegaTI(object):
    def __init__(self):
        pass

    def _eval(self, Ps, degree):
        n_features, n_components = Ps[0].shape
        cache = np.zeros((degree + 1, n_components))
        result = []
        for P in Ps:
            cache[:, :] = 0
            cache[0] = 1.0
            for j in range(n_features):
                abs_p_j = abs(P[j])
                for deg in range(degree):
                    cache[degree - deg] += cache[degree - deg - 1] * abs_p_j
            result.append(np.sum(cache[degree]))
        return np.array(result)

    def _eval_all(self, Ps):
        return np.sum(np.sum(np.prod(np.abs(Ps) + 1.0, axis=1), 0))

    def eval(self, P, degree):
        shape = P.shape[:-2]
        if degree > 0:
            result = self._eval(P.reshape(-1, P.shape[-2], P.shape[-1]), degree)
            if P.ndim == 2:
                return result[0]
            else:
                return result.reshape(shape)
        elif degree == -1:
            return self._eval_all(P.reshape(-1, P.shape[-2], P.shape[-1]))
        else:
            raise ValueError("degree must be a positive int or -1 (all).")

    def init_cache_pcd(self, degree, n_features, n_components):
        self._abs_p = np.zeros(n_features)
        if degree > 0:
            self._cache = np.zeros(degree + 1)
            self._dcache = np.zeros(degree + 1)
        elif degree == -1:
            self._cache_all_subsets = 1.0
        else:
            raise ValueError("degree must be a positive int or -1 (all).")

    def compute_cache_pcd_all(self, P, degree):
        pass

    def compute_cache_pcd(self, P, degree, s):
        if degree > 0:
            self._cache[0] = 1.0
            self._cache[1:] = 0.0
            self._dcache[:] = 0.0
            self._dcache[1] = 1.0
            for s, p_sj in enumerate(P[s]):
                abs_p_sj = abs(p_sj)
                self._abs_p[s] = abs_p_sj
                for deg in range(degree):
                    self._cache[degree - deg] += (
                        self._cache[degree - deg - 1] * abs_p_sj
                    )
        else:  # factorization machine
            self._cache_all_subsets = 1.0
            for j, p_sj in enumerate(P[s]):
                abs_p_sj = abs(p_sj)
                self._abs_p[j] = abs_p_sj
                self._cache_all_subsets *= 1.0 + abs_p_sj

    def update_cache_pcd(self, P, degree, s, j):
        abs_p_sj = abs(P[s, j])
        if degree > 0:  # factorization machine
            for deg in range(1, degree):
                self._cache[deg] = self._dcache[deg + 1] + self._dcache[deg] * abs_p_sj
        else:  # all-subsets
            self._cache_all_subsets *= 1.0 + abs_p_sj
        self._abs_p[j] = abs_p_sj

    def prox_cd(self, p_sj, strength, degree, j):
        sign = 1 if p_sj > 0 else -1
        if degree > 0:  # factorization machine
            for deg in range(2, degree + 1):
                self._dcache[deg] = self._cache[deg - 1]
                self._dcache[deg] -= self._dcache[deg - 1] * self._abs_p[j]
                if self._dcache[deg] < 0:
                    self._dcache[deg] = 0.0
            strength *= self._dcache[degree]
        else:  # all-subsets
            self._cache_all_subsets /= 1.0 + self._abs_p[j]
            strength *= self._cache_all_subsets

        return sign * np.maximum(abs(p_sj) - strength, 0)
