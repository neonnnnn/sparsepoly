# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
from numba import njit


@njit
def _precompute_A_all(X, P, A):
    n_features = X.get_n_features()
    n_components = P.shape[1]
    A[:, :] = 1.0
    for j in range(n_features):
        n_nz, indices, data = X.get_column(j)
        for ii in range(n_nz):
            i = indices[ii]
            x_ij = data[ii]
            for s in range(n_components):
                A[i, s] *= 1.0 + P[j, s] * x_ij


@njit
def _update(
    p_j,
    j,
    indices,
    data,
    n_nz,
    y,
    y_pred,
    loss,
    lams,
    beta,
    gamma,
    eta,
    A,
    regularizer,
    grad,
    inv_step_sizes,
):
    n_components = len(p_j)
    grad[:] = 0.0
    inv_step_sizes[:] = 0.0
    for ii in range(n_nz):
        i = indices[ii]
        x_ij = data[ii]
        dloss = loss.dloss(y_pred[i], y[i])
        for s in range(n_components):
            dA = x_ij * A[i, s] / (1.0 + x_ij * p_j[s])
            grad[s] += dloss * dA
            inv_step_sizes[s] += dA**2
    grad *= lams
    for s in range(n_components):
        grad[s] += beta * p_j[s]

    inv_step_size = 0
    for s in range(n_components):
        inv_step_size += inv_step_sizes[s]
    inv_step_size *= loss.mu
    inv_step_size += beta

    grad /= inv_step_size
    p_j -= eta * grad
    regularizer.prox_bcd(p_j, eta * gamma / inv_step_size, -1, j)


@njit
def pbcd_epoch(
    P,
    X,
    y,
    y_pred,
    lams,
    beta,
    gamma,
    eta,
    regularizer,
    loss,
    A,
    grad,
    inv_step_sizes,
    p_j_old,
    indices_feature,
):
    sum_viol = 0
    n_components = P.shape[1]
    # Update P_{s} \forall s \in [n_components] for A^{degree}
    # P_{s} for A^{degree} = P[order, s]
    # initialize the cached ds for this s
    _precompute_A_all(X, P, A)

    # update cache for sparse regularization
    regularizer.compute_cache_pbcd(P, -1)
    for j in indices_feature:
        n_nz, indices, data = X.get_column(j)
        # compute coordinate update
        p_j_old[:] = P[j]
        _update(
            P[j],
            j,
            indices,
            data,
            n_nz,
            y,
            y_pred,
            loss,
            lams,
            beta,
            gamma,
            eta,
            A,
            regularizer,
            grad,
            inv_step_sizes,
        )

        # synchronize predictions and A
        for ii in range(n_nz):
            i = indices[ii]
            x_ij = data[ii]
            y_pred[i] -= np.dot(lams, A[i])
            for s in range(n_components):
                A[i, s] /= 1.0 + x_ij * p_j_old[s]
                A[i, s] *= 1.0 + x_ij * P[j, s]
            y_pred[i] += np.dot(lams, A[i])
        regularizer.update_cache_pbcd(P, -1, j)

        p_j_old -= P[j]
        sum_viol += np.linalg.norm(p_j_old, 1)

    return sum_viol
