# encoding: utf-8
# Author: Kyohei Atarashi 
# License: MIT

import numpy as np
from numba import njit


@njit
def _precompute_A_all(X, p_s, A):
    n_features = X.get_n_features()
    A[:] = 1.0
    for j in range(n_features):
        n_nz, indices, data = X.get_column(j)
        p_sj = p_s[j]
        for ii in range(n_nz):
            i = indices[ii]
            x_ij = data[ii]
            A[i] *= (1.0 + p_sj * x_ij)


@njit
def _update(p_sj, j, indices, data, n_nz, y, y_pred, loss, lam,
            beta, gamma, eta, A, regularizer):
    update = 0
    inv_step_size = 0
    for ii in range(n_nz):
        i = indices[ii]
        x_ij = data[ii]
        dA = x_ij * A[i] / (1.0 + x_ij * p_sj)
        update += loss.dloss(y_pred[i], y[i]) * dA
        inv_step_size += dA ** 2
    
    inv_step_size *= loss.mu
    inv_step_size += beta

    update *= lam
    update += beta * p_sj
    update /= inv_step_size
    p_sj_new = p_sj - eta * update
    return regularizer.prox_cd(p_sj_new, eta*gamma/inv_step_size, -1, j)


@njit
def pcd_epoch(P, X, y, y_pred, lams, beta, gamma, eta, regularizer,
              loss, A, indices_component, indices_feature):
    sum_viol = 0
    regularizer.compute_cache_pcd_all(P, -1)
    for s in indices_component:
        # initialize the cached ds for this s
        _precompute_A_all(X, P[s], A)
        # compute cache for sparse regularization
        regularizer.compute_cache_pcd(P, -1, s)
        for j in indices_feature:
            n_nz, indices, data = X.get_column(j)
            # compute coordinate update
            p_sj_old = P[s, j]
            p_sj_new = _update(p_sj_old, j, indices, data, n_nz, y, y_pred,
                               loss, lams[s], beta, gamma, eta, A, regularizer)
            update = p_sj_old - p_sj_new
            sum_viol += abs(update)
            P[s, j] = p_sj_new

            # synchronize predictions and A
            for ii in range(n_nz):
                i = indices[ii]
                x_ij = data[ii]
                y_pred[i] -= lams[s] * A[i]
                A[i] /= (1.0 + x_ij * p_sj_old)
                A[i] *= (1.0 + x_ij * p_sj_new)
                y_pred[i] += lams[s] * A[i]
            # update cache for sparse regularizer
            regularizer.update_cache_pcd(P, -1, s, j)

    return sum_viol
