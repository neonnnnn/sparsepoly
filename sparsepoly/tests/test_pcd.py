# encoding: utf-8
# Author: Kyohei Atarashi 
# License: MIT

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sparsepoly.kernels import _poly_predict, all_subsets_kernel
from polylearn.kernels import anova_kernel
from sklearn.utils import check_random_state
import warnings
from sparsepoly import SparseAllSubsetsClassifier
from sparsepoly import SparseAllSubsetsRegressor
from sparsepoly import SparseFactorizationMachineClassifier
from sparsepoly import SparseFactorizationMachineRegressor
from .regularizer import L1Slow, SquaredL12Slow, OmegaTISlow
from sparsepoly import L1, SquaredL12, OmegaTI
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
regularizers = ["l1", "omegati"]

def _pcd_epoch_slow(P, X, y, loss, regularizer, lams, degree, beta, gamma,
                    eta0, indices_component, indices_feature, kernel):
    sum_viol = 0
    n_features = X.shape[1]
    for s in indices_component:
        p_s = P[s]
        for j in indices_feature:
            # compute prediction
            y_pred = _poly_predict(X, P, lams, kernel, degree=degree)

            # compute grad and inv_step_size
            x = X[:, j]
            notj_mask = np.arange(n_features) != j
            X_notj = X[:, notj_mask]
            ps_notj = np.atleast_2d(p_s[notj_mask])
            if kernel == "anova":
                grad_kernel = anova_kernel(ps_notj, X_notj, degree=degree-1)
            else:
                grad_kernel = all_subsets_kernel(ps_notj, X_notj)
            grad_kernel *= x
            grad_y = lams[s] * grad_kernel.ravel()
            inv_step_size = loss.mu * np.dot(grad_y, grad_y) + beta

            dloss = loss.dloss(y_pred, y)
            step = np.dot(dloss, grad_y) + beta * p_s[j]
            step /= inv_step_size

            # update
            p_sj_new = regularizer.prox_cd(
                p_s[j]-eta0*step, p_s, eta0*gamma/inv_step_size, degree, j
            )
            sum_viol += np.abs(p_sj_new - P[s, j])
            P[s, j] = p_sj_new

    return sum_viol


def pcd_slow(X, y, loss, regularizer, lams=None, degree=2, n_components=5,
             beta=1., gamma=1e-3, max_iter=10, tol=1e-5, eta0=1.0, verbose=False,
             random_state=None, mean=False, shuffle=False):

    n_samples, n_features = X.shape
    rng = check_random_state(random_state)
    P = 0.01 * rng.randn(n_components, n_features)
    if lams is None:
        lams = np.ones(n_components)

    if loss == "squared":
        loss = SquaredSlow()
    elif loss == "squared_hinge":
        loss = SquaredHingeSlow()
    elif loss == "logistic":
        loss = LogisticSlow()

    if regularizer == "l1":
        regularizer = L1Slow()
    elif regularizer == "squaredl12":
        regularizer = SquaredL12Slow()
    elif regularizer == "omegati":
        regularizer = OmegaTISlow()

    indices_feature = np.arange(n_features, dtype=np.int32)
    indices_component = np.arange(n_components, dtype=np.int32)
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
            rng.shuffle(indices_component)
            rng.shuffle(indices_feature)

        sum_viol = _pcd_epoch_slow(P, X, y, loss, regularizer, lams, degree,
                                   beta, gamma, eta0, indices_component,
                                   indices_feature, kernel)
        if verbose:
            print("Epoch", i, "violations", sum_viol)
        if sum_viol < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Objective did not converge. Increase max_iter.")

    return P


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2, 3, 4], [True, False], loss_reg, regularizers))
def test_fm_same_as_slow_reg(degree, mean, loss, regularizer):

    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    reg = SparseFactorizationMachineRegressor(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, shuffle=False) 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2], [True, False], loss_reg, ["squaredl12"]))
def test_fm_squaredl12_same_as_slow_reg(degree, mean, loss, regularizer):

    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    reg = SparseFactorizationMachineRegressor(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=1, random_state=0,
        mean=mean, shuffle=False) 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        print("")
        P_fit_slow = pcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=1, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2, 3, 4], [True, False], loss_clf, regularizers))
def test_fm_same_as_slow_clf(degree, mean, loss, regularizer):

    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)
    y = np.sign(y)

    reg = SparseFactorizationMachineClassifier(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, loss=loss, shuffle=False) 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("degree, mean, loss, regularizer", 
                         product([2], [True, False], loss_clf, ["squaredl12"]))
def test_fm_squaredl12_same_as_slow_clf(degree, mean, loss, regularizer):

    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)
    y = np.sign(y)

    reg = SparseFactorizationMachineClassifier(
        degree=degree, n_components=n_components, fit_lower=None,
        fit_linear=False, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, loss=loss, shuffle=False) 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=degree, n_components=n_components, beta=1, gamma=1e-3,
            max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


@pytest.mark.parametrize("mean, loss, regularizer", 
                         product([True, False], loss_reg, regularizers))
def test_all_subsets_same_as_slow_reg(mean, loss, regularizer):
    y = _poly_predict(X, P, lams, kernel="all-subsets")
    reg = SparseAllSubsetsRegressor(
        n_components=n_components, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, shuffle=False) 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=-1, n_components=n_components, beta=1, gamma=1e-3,
            eta0=0.1, max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_, P_fit_slow, decimal=4)


@pytest.mark.parametrize("mean, loss, regularizer", 
                         product([True, False], loss_clf, regularizers))
def test_all_subsets_same_as_slow_clf(mean, loss, regularizer):
    y = _poly_predict(X, P, lams, kernel="all-subsets")
    y = np.sign(y)

    reg = SparseAllSubsetsClassifier(
        n_components=n_components, beta=1, gamma=1e-3, regularizer=regularizer,
        warm_start=False, tol=1e-3, max_iter=5, random_state=0,
        mean=mean, loss=loss, shuffle=False) 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)
        P_fit_slow = pcd_slow(
            X, y, loss=loss, regularizer=regularizer, lams=reg.lams_, 
            degree=-1, n_components=n_components, beta=1, gamma=1e-3,
            eta0=0.1, max_iter=5, tol=1e-3, random_state=0, mean=mean) 
    assert_array_almost_equal(reg.P_, P_fit_slow, decimal=4)
