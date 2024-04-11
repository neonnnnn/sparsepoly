# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
from numba import njit


@njit
def _grad_anova(dA, A, x_ij, p_sj, degree):
    dA[0] = x_ij
    for t in range(1, degree):
        dA[t] = x_ij * (A[t] - p_sj * dA[t - 1])


@njit
def _precompute_A_all_degree(X, P, A, s, degree):
    n_features = X.get_n_features()

    A[:, 0] = 1.0
    A[:, 1:] = 0.0
    # calc {1, \ldots, degree}-order anova kernels for all data
    # A[m, i] = m-order anova kernel for i-th data
    for j in range(n_features):
        n_nz, indices, data = X.get_column(j)
        p_sj = P[s, j]
        for ii in range(n_nz):
            i = indices[ii]
            x_ij = data[ii]
            for t in range(degree):
                A[i, degree - t] += A[i, degree - t - 1] * p_sj * x_ij


@njit
def _update(
    p_sj,
    j,
    indices,
    data,
    n_nz,
    y,
    y_pred,
    loss,
    lam,
    degree,
    beta,
    gamma,
    eta,
    A,
    dA,
    regularizer,
):
    inv_step_size = 0
    update = 0
    for ii in range(n_nz):
        i = indices[ii]
        x_ij = data[ii]
        _grad_anova(dA, A[i], x_ij, p_sj, degree)
        update += loss.dloss(y_pred[i], y[i]) * dA[degree - 1]
        inv_step_size += dA[degree - 1] ** 2

    inv_step_size *= loss.mu
    inv_step_size += beta

    update *= lam
    update += beta * p_sj
    update /= inv_step_size
    p_sj_new = p_sj - eta * update
    return regularizer.prox_cd(p_sj_new, eta * gamma / inv_step_size, degree, j)


@njit
def pcd_epoch(
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
    indices_component,
    indices_feature,
):
    sum_viol = 0
    # Update P_{s} \forall s \in [n_components]
    regularizer.compute_cache_pcd_all(P, degree)
    for s in indices_component:
        # initialize the cached anova kernels for this s
        _precompute_A_all_degree(X, P, A, s, degree)
        # compute cache for sparse regularization
        regularizer.compute_cache_pcd(P, degree, s)
        for j in indices_feature:
            n_nz, indices, data = X.get_column(j)
            # compute coordinate update
            p_sj_old = P[s, j]
            p_sj_new = _update(
                p_sj_old,
                j,
                indices,
                data,
                n_nz,
                y,
                y_pred,
                loss,
                lams[s],
                degree,
                beta,
                gamma,
                eta,
                A,
                dA,
                regularizer,
            )
            update = p_sj_old - p_sj_new
            sum_viol += abs(update)
            P[s, j] = p_sj_new

            # synchronize predictions and caches
            for ii in range(n_nz):
                i = indices[ii]
                x_ij = data[ii]
                dA[0] = x_ij
                for deg in range(1, degree):
                    dA[deg] = x_ij * (A[i, deg] - p_sj_old * dA[deg - 1])
                    A[i, deg] -= update * dA[deg - 1]

                A[i, degree] -= update * dA[degree - 1]
                y_pred[i] -= lams[s] * update * dA[degree - 1]
            # update cache for sparse regularizer
            regularizer.update_cache_pcd(P, degree, s, j)

    return sum_viol
