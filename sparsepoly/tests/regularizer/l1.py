import numpy as np

from .utils import norm, soft_thresholding


class L1Slow(object):
    def __init__(self):
        pass

    def eval(self, P):
        return norm(P, ord=1, axis=(-2, -1))

    def prox_cd(self, p_sj, ps, strength, degree, j):
        return np.sign(p_sj) * max(abs(p_sj) - strength, 0.0)

    def prox_bcd(self, P, strength, degree, j):
        soft_thresholding(P[j], strength)

    def prox(self, P, strength, degree):
        soft_thresholding(P, strength)
