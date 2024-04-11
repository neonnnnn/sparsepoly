from random import randint

import numpy as np
from numba import jit, njit


@njit
def soft_thresholding(x, y):
    x[:] = np.sign(x) * np.maximum(np.abs(x)-y, 0.0)


@njit
def norm(X, ord=2, axis=None):
    abs_X = np.abs(X)
    result = np.sum(abs_X ** ord, axis=axis)
    return result ** (1/ord)


@njit
def swap(x, i, j):
    tmp = x[i]
    x[i] = x[j]
    x[j] = tmp


@njit
def prox_squaredl12(p, strength, candidates):
    S = 0.0
    theta = 0
    offset = 0
    n = len(p)
    n_candidates = len(p)
    for i in range(n):
        candidates[i] = i
    while n_candidates > 0:
        ii = randint(0, n_candidates-1)
        i = candidates[offset+ii]
        pivot = abs(p[i])
        swap(candidates, offset+ii, offset+n_candidates-1)
        
        n_greater = 1
        n_lower = 0
        S_Gi = pivot

        for ii2 in range(n_candidates-1):
            i2 = candidates[offset+ii2]
            if pivot > abs(p[i2]):
                swap(candidates, offset+ii2, offset+n_lower)
                n_lower += 1
            else:
                n_greater += 1
                S_Gi += abs(p[i2])
            
        cond = 2 * strength * (S+S_Gi) / (1.0+2.0*strength*(theta+n_greater))
        if pivot >= cond:
            # offset = offset
            n_candidates = n_lower
            S += S_Gi
            theta += n_greater
        else: # G
            offset += n_lower
            n_candidates = 0
            for ii2 in range(n_greater-1):
                i2 = candidates[offset+ii2]
                if pivot < abs(p[i2]):
                    # swap
                    swap(candidates, offset+ii2, offset+n_candidates)
                    n_candidates += 1
    S /= 1.0 + 2.0*strength*theta
    soft_thresholding(p, 2*strength*S)
