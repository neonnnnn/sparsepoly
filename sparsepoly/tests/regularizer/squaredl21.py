from math import sqrt

import numpy as np

from .utils import norm, prox_squaredl12


class SquaredL21Slow(object):
    def __init__(self, transpose=False):
        self.transpose = transpose

    def eval(self, P):
        axis = -2 if self.transpose else -1
        return norm(norm(P, ord=2, axis=axis), ord=1, axis=-1) ** 2

    def prox_bcd(self, P, strength, degree, j):
        P[j] /= 1 + 2 * strength
        l2 = sqrt(np.dot(P[j], P[j]))
        norms = norm(P, 2, axis=1)
        norms[j] = 0.0
        strength = 2 * np.sum(norms) * strength / (1.0 + 2 * strength)
        if l2 > strength:
            P[j] *= 1.0 - strength / l2
        else:
            P[j] = 0.0

    def prox(self, P, strength, degree):
        axis = 0 if self.transpose else 1
        norms = norm(P, 2, axis)
        prox_squaredl12(norms, strength)
        norms[norms <= strength] = np.inf
        norms = np.expand_dims(norms, axis=axis)
        P *= 1.0 - strength / norms
