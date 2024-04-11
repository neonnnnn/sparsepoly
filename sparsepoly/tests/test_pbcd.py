# encoding: utf-8
# Author: Kyohei Atarashi 
# License: MIT

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sparsepoly.kernels import poly_predict, all_subsets_kernel
from sparsepoly.kernels import anova_kernel
from sklearn.utils import check_random_state
import warnings
from sparsepoly import SparseAllSubsetsClassifier
from sparsepoly import SparseAllSubsetsRegressor
from sparsepoly import SparseFactorizationMachineClassifier
from sparsepoly import SparseFactorizationMachineRegressor
from .regularizer import L21Slow, SquaredL21Slow, OmegaCSSlow, L1Slow
from .loss_slow import SquaredSlow, SquaredHingeSlow, LogisticSlow

from itertools import product

n_components = 5
n_features = 4
n_samples = 20

rng = np.random.RandomState(1)

X = rng.randn(n_samples, n_features)
P = rng.randn(n_components, n_features)

lams = rng.randn(n_components)

loss_reg = ["squared"]
loss_clf = ["squared_hinge", "logistic"]
regularizers = ["l1", "l21", "omegacs"]


def pbcd_epoch_slow(P, X, y, loss, regularizer, lams, degree, beta, gamma,
                     eta0, indices_feature, kernel):
    sum_viol = 0
    n_features = X.shape[1]
    for j in indices_feature:
        # compute prediction
        y_pred = poly_predict(X, P.T, lams, kernel, degree=degree)

        # compute grad and inv_step_size
        x = X[:, j]
        notj_mask = np.arange(n_features) != j
        X_notj = X[:, notj_mask]
        P_notj = P[notj_mask]
        if kernel == "anova":
            grad_kernel = anova_kernel(P_notj.T, X_notj, degree=degree-1)
        else:
            grad_kernel = all_subsets_kernel(P_notj.T, X_notj)
        grad_kernel *= x # (n_components, n_samples)
        grad_y = grad_kernel * lams[:, None]
        l2_reg = beta
        inv_step_size = loss.mu * np.sum(grad_y*grad_y) + l2_reg

        dloss = loss.dloss(y_pred, y)
        step = np.sum(dloss*grad_y, axis=1) + l2_reg * P[j]
        step /= inv_step_size

        # update
        p_j_old = np.array(P[j])
        P[j] -= eta0 * step
        regularizer.prox_bcd(
            P, eta0*gamma/inv_step_size, degree, j
        )
        sum_viol += np.sum(np.abs(p_j_old - P[j]))

    return sum_viol


def pbcd_slow(X, y, loss, regularizer, lams=None, degree=2, n_components=5,
              beta=1., gamma=1e-3, max_iter=10, tol=1e-5, eta0=1.0, verbose=False,
              random_state=None, mean=False, shuffle=False):

    n_samples, n_features = X.shape
    rng = check_random_state(random_state)
    P = 0.01 * rng.randn(n_components, n_features)
    P = np.array(P.T)
    if lams is None:
        lams = np.ones(n_components)

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
    elif regularizer == "omegacs":
        regularizer = OmegaCSSlow()

    indices_feature = np.arange(n_features, dtype=np.int32)
    converged = False
    if mean:
        beta = beta * n_samples
        gamma = gamma * n_samples
    else:
        beta = beta
        gamma = gamma

    converged = False
    if degree == -1:
        kernel = "all-subsets"
    else:
        kernel = "anova"

    # start optimization
    for i in range(max_iter):
        if shuffle:
            rng.shuffle(indices_feature)

        sum_viol = pbcd_epoch_slow(P, X, y, loss, regularizer, lams, degree,
                                    beta, gamma, eta0, indices_feature, kernel)
        if verbose:
            print("Epoch", i, "violations", sum_viol)
        if sum_viol < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Objective did not converge. Increase max_iter.")

    return P.T


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2, 3, 4], [True, False], loss_reg, regularizers))
def test_fm_same_as_slow_reg(degree, mean, loss, regularizer):

    y = poly_predict(X, P, lams, kernel="anova", degree=degree)

    reg = SparseFactorizationMachineRegressor(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, shuffle=False, solver="pbcd") 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = P
        P_fit_slow = pbcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2], [True, False], loss_reg, ["squaredl21"]))
def test_fm_squaredl21_same_as_slow_reg(degree, mean, loss, regularizer):

    y = poly_predict(X, P, lams, kernel="anova", degree=degree)

    reg = SparseFactorizationMachineRegressor(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=1, random_state=0,
        mean=mean, shuffle=False, solver="pbcd") 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        print("")
        P_fit_slow = pbcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=1, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2, 3, 4], [True, False], loss_clf, regularizers))
def test_fm_same_as_slow_clf(degree, mean, loss, regularizer):

    y = poly_predict(X, P, lams, kernel="anova", degree=degree)
    y = np.sign(y)

    reg = SparseFactorizationMachineClassifier(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, loss=loss, shuffle=False, solver="pbcd") 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pbcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2], [True, False], loss_clf, ["squaredl21"]))
def test_fm_squaredl21_same_as_slow_clf(degree, mean, loss, regularizer):

    y = poly_predict(X, P, lams, kernel="anova", degree=degree)
    y = np.sign(y)

    reg = SparseFactorizationMachineClassifier(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, loss=loss, shuffle=False, solver="pbcd") 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pbcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("mean, loss, regularizer", 
                         product([True, False], loss_reg, regularizers))
def test_all_subsets_same_as_slow_reg(mean, loss, regularizer):
    y = poly_predict(X, P, lams, kernel="all-subsets")
    reg = SparseAllSubsetsRegressor(
        n_components=n_components, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, shuffle=False, solver="pbcd") 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pbcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=-1, n_components=n_components, beta=1, gamma=1e-3,
            eta0=0.1, max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_, P_fit_slow, decimal=4)


@pytest.mark.parametrize("mean, loss, regularizer", 
                         product([True, False], loss_clf, regularizers))
def test_all_subsets_same_as_slow_clf(mean, loss, regularizer):
    y = poly_predict(X, P, lams, kernel="all-subsets")
    y = np.sign(y)

    reg = SparseAllSubsetsClassifier(
        n_components=n_components, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, loss=loss, shuffle=False, solver="pbcd") 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pbcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=-1, n_components=n_components, beta=1, gamma=1e-3,
            eta0=0.1, max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_, P_fit_slow, decimal=4)
