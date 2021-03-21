# encoding: utf-8
# Author: Kyohei Atarashi 
# License: MIT

import numpy as np
from numba import njit


@njit
def _get_eta(learning_rate, eta0, alpha, beta, power_t, it):
    if learning_rate == 0: # constant
        return eta0, eta0
    elif learning_rate == 1: # optimal
        eta_it = eta0 * it
        eta_P = eta0 / ((1.0 + eta_it * beta) ** power_t)
        eta_w = eta0 / ((1.0 + eta_it * alpha) ** power_t)
        return eta_P, eta_w
    elif learning_rate == 2: # pegasos
        return 1.0 / (beta*it), 1.0 / (alpha*it)
    elif learning_rate == 3: # invscaling
        eta = eta0 / (it ** power_t)
        return eta, eta


@njit
def _grad_anova(dA, A, x_ij, p_j, degree):
    dA[0] = x_ij
    n_components = A.shape[1] # = dA.shape[1]
    for t in range(1, degree):
        for s in range(n_components):
            dA[t, s] = x_ij * (A[t, s] - p_j[s] * dA[t-1, s])


@njit
def _anova(indices, data, n_nz, P, A, degree):
    A[0] = 1.0
    A[1:] = 0.0
    n_components = A.shape[1]
    for jj in range(n_nz):
        j = indices[jj]
        x_ij = data[jj]
        for t in range(degree):
            for s in range(n_components):
                A[degree-t, s] += A[degree-t-1, s] * x_ij * P[j, s]


@njit
def _pred(indices, data, n_nz, P, w, lams, A, degree):
    y_pred = 0.0
    for jj in range(n_nz):
        j = indices[jj]
        y_pred += data[jj] * w[j]
    for order in range(P.shape[0]):
        deg = degree - order
        _anova(indices, data, n_nz, P[order], A[order], deg)
        y_pred += np.dot(lams, A[order, deg])
    return y_pred
        

@njit
def _update_grads(indices, data, n_nz, P, lams, A, dA, grad_P, grad_w, degree,
                  yi, y_pred, loss, fit_linear):
    dL = loss.dloss(y_pred, yi)
    if fit_linear:
        for jj in range(n_nz):
            j = indices[jj]
            x_ij = data[jj]
            grad_w[j] += dL * x_ij
    n_components = P.shape[2]
    for order in range(P.shape[0]):
        deg = degree - order
        for jj in range(n_nz):
            j = indices[jj]
            x_ij = data[jj]
            _grad_anova(dA, A[order], x_ij, P[order, j], deg)
            for s in range(n_components):
                grad_P[order, j, s] += dL * lams[s] * dA[deg-1, s]


@njit
def _update_params(P, w, grad_P, grad_w, alpha, beta, gamma, eta_P, eta_w,
                   degree, fit_linear, regularizer, batch_size):
    # sgd update
    if fit_linear:
        grad_w *= eta_w / batch_size
        w -= grad_w
        w /= 1 + eta_w * alpha
    grad_P *= eta_P / batch_size
    P -= grad_P
    P /= 1.0 + eta_P*beta

    # proximal update
    for order in range(len(P)):
        deg = degree - order
        regularizer.prox(P[order], gamma * eta_P / (1+eta_P*beta), deg)
    

@njit
def _psgd_epoch(X, y, P, w, lams, degree, alpha, beta, gamma, regularizer, loss, A,
                dA, grad_P, grad_w, indices_samples, fit_linear,
                eta0, learning_rate, power_t, batch_size, it):
    n_samples = X.get_n_samples()
    b = 0
    sum_loss = 0.0
    for ii, i in enumerate(indices_samples):
        # compute prediction and dA
        n_nz, indices, data = X.get_row(i)
        y_pred = _pred(indices, data, n_nz, P, w, lams, A, degree)
        sum_loss += loss.loss(y_pred, y[i])
        # update gradients
        _update_grads(indices, data, n_nz, P, lams, A, dA, grad_P,
                      grad_w, degree, y[i], y_pred, loss, fit_linear)
        b += 1
        # update parameters if the batch is filled
        if b == batch_size or ii == (n_samples - 1):
            eta_P, eta_w = _get_eta(learning_rate, eta0, alpha, beta, power_t, it)
            _update_params(
                P, w, grad_P, grad_w, alpha, beta, gamma,
                eta_P, eta_w, degree, fit_linear, regularizer, b
            )
            
            grad_P[:] = 0.0
            grad_w[:] = 0.0
            b = 0
            it += 1
    return sum_loss, it
