import numpy as np

from .utils import norm, prox_squaredl12


class SquaredL12Slow(object):
    def __init__(self, transpose=True):
        self.transpose = transpose

    def eval(self, P):
        axis = -2 if self.transpose else -1
        return norm(norm(P, ord=1, axis=axis), axis=-1) ** 2

    def prox_cd(self, p_sj, p_s, strength, degree, j):
        dcache = (np.sum(np.abs(p_s)) - abs(p_s[j]))
        sign = 1 if p_sj > 0 else -1
        return sign * max(abs(p_sj) - 2*strength*dcache, 0) / (1+2*strength)

    def prox(self, P, strength, degree):
        n_features, n_components = P.shape
        if self.transpose:
            for s in range(n_components):
                cache = P[:, s]
                prox_squaredl12(cache, strength)
                P[:, s] = cache
        else:
            for j in range(n_features):
                prox_squaredl12(P[j], strength)
