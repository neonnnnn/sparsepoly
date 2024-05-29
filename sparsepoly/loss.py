# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from math import exp, log

from numba import float64
from numba.experimental import jitclass

spec = [("mu", float64)]


@jitclass(spec)
class Squared(object):
    """Squared loss: L(p, y) = 0.5 * (y - p)²"""

    def __init__(self):
        self.mu = 1

    def loss(self, p, y):
        return 0.5 * (p - y) ** 2

    def dloss(self, p, y):
        return p - y


@jitclass(spec)
class Logistic(object):
    """Logistic loss: L(p, y) = log(1 + exp(-yp))"""

    def __init__(self):
        self.mu = 0.25

    def loss(self, p, y):
        z = p * y
        # log(1 + exp(-z))
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z
        return log(1.0 + exp(-z))

    def dloss(self, p, y):
        z = p * y
        # def tau = 1 / (1 + exp(-z))
        # return y * (tau - 1)
        if z > 18.0:
            return -y * exp(-z)
        if z < -18.0:
            return -y
        return -y / (exp(z) + 1.0)


@jitclass(spec)
class SquaredHinge(object):
    """Squared hinge loss: L(p, y) = max(1 - yp, 0)²"""

    def __init__(self):
        self.mu = 2

    def loss(self, p, y):
        z = 1 - p * y
        if z > 0:
            return z * z
        return 0.0

    def dloss(self, p, y):
        z = 1 - p * y
        if z > 0:
            return -2 * y * z
        return 0.0


REGRESSION_LOSSES = {"squared": Squared()}

CLASSIFICATION_LOSSES = {
    "squared": Squared(),
    "squared_hinge": SquaredHinge(),
    "logistic": Logistic(),
}
