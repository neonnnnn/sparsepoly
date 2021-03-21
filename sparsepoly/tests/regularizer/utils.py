import numpy as np
from random import randint


def soft_thresholding(x, y):
    x[:] = np.sign(x) * np.maximum(np.abs(x)-y, 0.0)


def norm(X, ord=2, axis=None):
    abs_X = np.abs(X)
    result = np.sum(abs_X ** ord, axis=axis)
    return result ** (1/ord)


def prox_squaredl12(p, strength):
    S = 0.0
    theta = 0
    offset = 0
    n = len(p)
    n_candidates = len(p)
    candidates = np.arange(n, dtype=np.int32)
    
    while n_candidates > 0:
        ii = randint(0, n_candidates-1)
        i = candidates[offset+ii]
        pivot = abs(p[i])
        tmp = candidates[offset+ii]
        candidates[offset+ii] = candidates[offset+n_candidates-1]
        candidates[offset+n_candidates-1] = tmp
        n_greater = 1
        n_lower = 0
        S_Gi = pivot

        for ii2 in range(n_candidates-1):
            i2 = candidates[ii2]
            if pivot > abs(p[i2]):
                tmp = candidates[offset+ii2]
                candidates[offset+ii2] = candidates[offset+n_lower]
                candidates[offset*n_lower] = tmp
                n_lower += 1
            else:
                n_greater += 1
                S_Gi += abs(p[i2])
            
        cond = 2 * strength * (S+S_Gi) / (1.0+2.0*strength*(theta+n_greater))
        if pivot > cond:
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
                    tmp = candidates[offset+ii2]
                    candidates[offset+ii2]  = candidates[offset+n_candidates]
                    candidates[offset+n_candidates] = tmp
                    n_candidates += 1
    S /= 1.0 + 2.0*strength*theta
    soft_thresholding(p, 2*strength*S)
