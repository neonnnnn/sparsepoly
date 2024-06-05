# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
from numba import njit


@njit
def _grad_anova(dA, A, x_ij, p_j, degree):
    n_components = len(p_j)
    dA[0] = x_ij
    for t in range(1, degree):
        for s in range(n_components):
            dA[t, s] = x_ij * (A[t, s] - p_j[s] * dA[t - 1, s])


@njit
def _precompute_A_all_degree(X, P, A, degree):
    n_features = X.get_n_features()
    n_components = P.shape[1]
    A[:, 0] = 1.0
    A[:, 1:] = 0.0
    # calc {1, \ldots, degree}-order anova kernels for all data
    # A[m, i] = m-order anova kernel for i-th data
    for j in range(n_features):
        n_nz, indices, data = X.get_column(j)
        for ii in range(n_nz):
            i = indices[ii]
            x_ij = data[ii]
            for t in range(degree):
                for s in range(n_components):
                    A[i, degree - t, s] += A[i, degree - t - 1, s] * P[j, s] * x_ij


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
    degree,
    beta,
    gamma,
    eta,
    A,
    dA,
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
        _grad_anova(dA[i], A[i], x_ij, p_j, degree)
        dloss = loss.dloss(y_pred[i], y[i])
        for s in range(n_components):
            grad[s] += dloss * dA[i, degree - 1, s]
            inv_step_sizes[s] += dA[i, degree - 1, s] ** 2
    inv_step_size = 0
    for s in range(n_components):
        inv_step_size += inv_step_sizes[s]
    inv_step_size *= loss.mu
    inv_step_size += beta

    grad *= lams
    for s in range(n_components):
        grad[s] += beta * p_j[s]
    grad /= inv_step_size
    p_j -= eta * grad
    regularizer.prox_bcd(p_j, eta * gamma / inv_step_size, degree, j)


@njit
def pbcd_epoch(
    P,
    X,
    y,
    y_pred,
    lams,
    degree,
    beta,
    gamma,
    eta,
    regularizer,
    loss,
    A,
    dA,
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
    _precompute_A_all_degree(X, P, A, degree)
    # update cache for sparse regularization
    regularizer.compute_cache_pbcd(P, degree)
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
            degree,
            beta,
            gamma,
            eta,
            A,
            dA,
            regularizer,
            grad,
            inv_step_sizes,
        )

        # synchronize predictions and caches
        updates = p_j_old
        updates -= P[j]
        for ii in range(n_nz):
            i = indices[ii]
            for deg in range(1, degree + 1):
                for s in range(n_components):
                    A[i, deg, s] -= updates[s] * dA[i, deg - 1, s]
            for s in range(n_components):
                y_pred[i] -= lams[s] * updates[s] * dA[i, degree - 1, s]
        regularizer.update_cache_pbcd(P, degree, j)
        sum_viol += np.linalg.norm(updates, 1)

    return sum_viol
