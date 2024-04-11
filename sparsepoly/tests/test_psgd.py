# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import math
import warnings
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils import check_random_state

from sparsepoly import (
    SparseAllSubsetsClassifier,
    SparseAllSubsetsRegressor,
    SparseFactorizationMachineClassifier,
    SparseFactorizationMachineRegressor,
)
from sparsepoly.kernels import all_subsets_kernel, anova_kernel, poly_predict

from .loss_slow import LogisticSlow, SquaredHingeSlow, SquaredSlow
from .regularizer import L1Slow, L21Slow, SquaredL12Slow, SquaredL21Slow

n_components = 5
n_features = 4
n_samples = 50

rng = np.random.RandomState(1)

X = rng.randn(n_samples, n_features)
P = rng.randn(n_components, n_features)

lams = rng.randn(n_components)

loss_reg = ["squared"]
loss_clf = ["squared_hinge", "logistic"]
regularizers = ["l1", "l21", "squaredl12", "squaredl21"]


LEARNING_RATE = {"constant": 0, "optimal": 1, "pegasos": 2, "invscaling": 3}


def _get_eta(learning_rate, eta0, alpha, beta, power_t, it):
    if learning_rate == 0:  # constant
        return eta0, eta0
    elif learning_rate == 1:  # optimal
        eta_it = eta0 * it
        eta_P = eta0 / ((1.0 + eta_it * beta) ** power_t)
        eta_w = eta0 / ((1.0 + eta_it * alpha) ** power_t)
        return eta_P, eta_w
    elif learning_rate == 2:  # pegasos
        return 1.0 / (beta * it), 1.0 / (alpha * it)
    elif learning_rate == 3:  # invscaling
        eta = eta0 / (it**power_t)
        return eta, eta


def psgd_epoch_slow(
    P,
    w,
    X,
    y,
    loss,
    regularizer,
    lams,
    degree,
    alpha,
    beta,
    gamma,
    indices_samples,
    fit_linear,
    eta0,
    learning_rate,
    power_t,
    batch_size,
    it,
    kernel,
):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    sum_loss = 0.0
    n_minibatches = math.ceil(n_samples / batch_size)
    for ii in range(n_minibatches):
        # pick a minibatch
        minibatch_indices = np.atleast_1d(
            indices_samples[ii * batch_size : (ii + 1) * batch_size]
        )
        X_batch = X[minibatch_indices]
        y_batch = y[minibatch_indices]
        # compute prediction and loss
        y_pred_batch = poly_predict(X_batch, P.T, lams, kernel, degree=degree)
        y_pred_batch += np.dot(X_batch, w)
        sum_loss += np.sum(
            loss.loss(np.atleast_1d(y_pred_batch), np.atleast_1d(y_batch))
        )

        # compute grad and inv_step_size
        dloss = loss.dloss(np.atleast_1d(y_pred_batch), np.atleast_1d(y_batch))
        grad_P = np.zeros(P.shape)  # (n_features, n_components)
        for j in range(n_features):
            notj_mask = np.arange(n_features) != j
            X_batch_notj = X_batch[:, notj_mask]
            P_notj = P[notj_mask]
            # grad_kernel: (n_components, n_samples)
            if kernel == "anova":
                grad_kernel = anova_kernel(P_notj.T, X_batch_notj, degree=degree - 1)
            else:
                grad_kernel = all_subsets_kernel(P_notj.T, X_batch_notj)
            grad_P[j] = np.dot(
                grad_kernel, dloss * X_batch[:, j]
            )  # (n_components, n_samples)
        grad_P *= lams
        grad_P /= len(minibatch_indices)

        eta_P, eta_w = _get_eta(learning_rate, eta0, alpha, beta, power_t, it)
        P -= eta_P * grad_P
        P /= 1.0 + eta_P * beta
        # update
        regularizer.prox(
            P,
            eta_P * gamma / (1.0 + eta_P * beta),
            degree,
        )
        if fit_linear:
            grad_w = np.dot(X_batch.T, dloss) / len(minibatch_indices)
            w -= eta_w * grad_w
            w /= 1.0 + eta_w * alpha
        it += 1

    return sum_loss, it


def psgd_slow(
    X,
    y,
    loss,
    regularizer,
    lams=None,
    degree=2,
    n_components=5,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    max_iter=10,
    tol=1e-5,
    eta0=1.0,
    fit_linear=True,
    verbose=False,
    random_state=None,
    mean=False,
    shuffle=False,
    batch_size="auto",
    learning_rate=1,
    power_t=1.0,
    n_iter_no_change=5,
):

    n_samples, n_features = X.shape
    rng = check_random_state(random_state)
    P = 0.01 * rng.randn(n_components, n_features)
    P = np.array(P.T)
    w = np.zeros(n_features)

    if lams is None:
        lams = np.ones(n_components)

    if batch_size == "auto":
        batch_size = int(n_samples * n_features / np.count_nonzero(X))
    else:
        batch_size = batch_size
    if loss == "squared":
        loss = SquaredSlow()
    elif loss == "squared_hinge":
        loss = SquaredHingeSlow()
    elif loss == "logistic":
        loss = LogisticSlow()

    if regularizer == "l21":
        regularizer = L21Slow()
    elif regularizer == "l1":
        regularizer = L1Slow()
    elif regularizer == "squaredl21":
        regularizer = SquaredL21Slow()
    elif regularizer == "squaredl12":
        regularizer = SquaredL12Slow()

    indices_samples = np.arange(n_samples, dtype=np.int32)
    converged = False
    if degree == -1:
        kernel = "all-subsets"
    else:
        kernel = "anova"

    best_loss = np.inf
    no_improvement_count = 0
    # start optimization
    it = 1
    for epoch in range(max_iter):
        if shuffle:
            rng.shuffle(indices_samples)

        sum_loss, it = psgd_epoch_slow(
            P,
            w,
            X,
            y,
            loss,
            regularizer,
            lams,
            degree,
            alpha,
            beta,
            gamma,
            indices_samples,
            fit_linear,
            eta0,
            LEARNING_RATE[learning_rate],
            power_t,
            batch_size,
            it,
            kernel,
        )

        sum_loss /= n_samples
        if verbose:
            print("Epoch", epoch + 1, "loss", sum_loss)
        if sum_loss > (best_loss - tol):
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if sum_loss < best_loss:
            best_loss = sum_loss
        if no_improvement_count >= n_iter_no_change:
            converged = True
            break

    if not converged:
        warnings.warn("Objective did not converge. Increase max_iter.")

    return P.T, w


@pytest.mark.parametrize(
    "degree, batch_size, learning_rate, fit_linear, loss, regularizer",
    product(
        [2, 3, 4],
        [1, 5, 8, n_samples, "auto"],
        ["constant", "optimal"],
        [True, False],
        loss_reg,
        regularizers,
    ),
)
def test_fm_same_as_slow_reg(
    degree, batch_size, learning_rate, fit_linear, loss, regularizer
):

    y = poly_predict(X, P, lams, kernel="anova", degree=degree)

    reg = SparseFactorizationMachineRegressor(
        degree=degree,
        n_components=n_components,
        fit_lower=None,
        fit_linear=fit_linear,
        alpha=1e-3,
        beta=1e-3,
        gamma=0.0,
        regularizer=regularizer,
        learning_rate=learning_rate,
        eta0=0.01,
        warm_start=False,
        tol=1e-3,
        max_iter=10,
        random_state=0,
        shuffle=False,
        solver="psgd",
        batch_size=batch_size,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)
        P_fit_slow, w_fit_slow = psgd_slow(
            X,
            y,
            loss=loss,
            regularizer=regularizer,
            lams=reg.lams_,
            degree=degree,
            n_components=n_components,
            alpha=1e-3,
            beta=1e-3,
            gamma=0.0,
            learning_rate=learning_rate,
            eta0=0.01,
            shuffle=False,
            max_iter=10,
            tol=1e-3,
            random_state=0,
            fit_linear=fit_linear,
            batch_size=batch_size,
        )
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)
    assert_array_almost_equal(reg.w_, w_fit_slow, decimal=4)


@pytest.mark.parametrize(
    "degree, batch_size, learning_rate, fit_linear, loss, regularizer",
    product(
        [2, 3, 4],
        [1, 5, 8, n_samples, "auto"],
        ["constant", "optimal"],
        [True, False],
        loss_clf,
        regularizers,
    ),
)
def test_fm_same_as_slow_clf(
    degree, batch_size, learning_rate, fit_linear, loss, regularizer
):

    y = poly_predict(X, P, lams, kernel="anova", degree=degree)
    y = np.sign(y)

    reg = SparseFactorizationMachineClassifier(
        degree=degree,
        n_components=n_components,
        fit_lower=None,
        fit_linear=fit_linear,
        alpha=1e-3,
        beta=1e-3,
        gamma=0.0,
        regularizer=regularizer,
        learning_rate=learning_rate,
        eta0=0.01,
        warm_start=False,
        tol=1e-3,
        max_iter=10,
        random_state=0,
        shuffle=False,
        solver="psgd",
        batch_size=batch_size,
        verbose=0,
        loss=loss,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)
        P_fit_slow, w_fit_slow = psgd_slow(
            X,
            y,
            loss=loss,
            regularizer=regularizer,
            lams=reg.lams_,
            degree=degree,
            n_components=n_components,
            alpha=1e-3,
            beta=1e-3,
            gamma=0.0,
            learning_rate=learning_rate,
            eta0=0.01,
            shuffle=False,
            max_iter=10,
            tol=1e-3,
            random_state=0,
            fit_linear=fit_linear,
            batch_size=batch_size,
            verbose=0,
        )
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)
    assert_array_almost_equal(reg.w_, w_fit_slow, decimal=4)
