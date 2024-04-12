from math import sqrt

import numpy as np

from .utils import norm


class OmegaCSSlow(object):
    def __init__(self):
        pass

    def _eval(self, Ps, degree):
        n = len(Ps)
        n_features = Ps.shape[1]
        cache = np.zeros((degree + 1, n))
        cache[0] = 1.0
        norms = np.asfortranarray(norm(Ps, ord=2, axis=-1))
        for j in range(n_features):
            for deg in range(degree):
                cache[degree - deg] += cache[degree - deg - 1] * norms[:, j]
        return cache[-1]

    def eval(self, P, degree):
        shape = P.shape[:-2]
        result = self._eval(P.reshape(-1, P.shape[-1], P.shape[-2]), degree)
        if P.ndim == 2:
            return result[0]
        else:
            return result.reshape(shape)

    def prox_bcd(self, P, strength, degree, j):
        norms = norm(P, 2, axis=1)
        n_features = len(P)
        l2 = norms[j]
        norms[j] = 0.0
        if degree > 0:  # factorization machine
            cache = np.zeros(degree)
            cache[0] = 1
            for j2 in range(n_features):
                for t in range(degree - 1):
                    cache[degree - t - 1] += cache[degree - t - 2] * norms[j2]
            strength *= cache[-1]
        else:  # all-subsets
            cache = 1.0
            for j2 in range(n_features):
                cache *= 1.0 + norms[j2]
            strength *= cache

        if l2 > strength:
            P[j] *= 1 - strength / l2
        else:
            P[j] = 0.0
