from math import sqrt

import numpy as np

from .utils import norm


class L21Slow(object):
    def __init__(self, transpose=False):
        self.transpose = transpose

    def eval(self, P):
        axis = -2 if self.transpose else -1
        return norm(norm(P, ord=2, axis=axis), ord=1, axis=-1)

    def prox_bcd(self, P, strength, degree, j):
        l2 = sqrt(np.dot(P[j], P[j]))
        if l2 > strength:
            P[j] *= 1.0 - strength / l2
        else:
            P[j] = 0.0

    def prox(self, P, strength, degree):
        axis = 0 if self.transpose else 1
        norms = norm(P, 2, axis=axis)
        norms[norms <= strength] = np.inf
        P *= 1.0 - strength / np.expand_dims(norms, axis=axis)
