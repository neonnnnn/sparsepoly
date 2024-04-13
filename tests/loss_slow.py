# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from abc import ABCMeta, abstractmethod

import numpy as np


class SquaredSlow(object):
    """Squared loss: L(p, y) = 0.5 * (y - p)²"""

    def __init__(self):
        self.mu = 1

    def loss(self, p, y):
        return 0.5 * (p - y) ** 2

    def dloss(self, p, y):
        return p - y


class LogisticSlow(object):
    """Logistic loss: L(p, y) = log(1 + exp(-yp))"""

    def __init__(self):
        self.mu = 0.25

    def loss(self, p, y):
        z = p * y
        # log(1 + exp(-z))
        result = np.log(1.0 + np.exp(-z))
        result[z < -18.0] = -z[z < -18.0]
        result[z > 18.0] = np.exp(-z[z > 18.0])
        return result

    def dloss(self, p, y):
        z = p * y
        # def tau = 1 / (1 + exp(-z))
        # return y * (tau - 1)
        result = -y / (np.exp(z) + 1.0)
        result[z > 18.0] = -(y * np.exp(-z))[z > 18.0]
        result[z < -18.0] = -y[z < -18.0]
        return result


class SquaredHingeSlow(object):
    """Squared hinge loss: L(p, y) = max(1 - yp, 0)²"""

    def __init__(self):
        self.mu = 2

    def loss(self, p, y):
        return np.maximum(1 - p * y, 0) ** 2

    def dloss(self, p, y):
        z = 1 - p * y
        return -2 * y * np.maximum(z, 0.0)
