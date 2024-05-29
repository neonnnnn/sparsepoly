import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sparsepoly.regularizer import SquaredL12, SquaredL21

n_features = 2000
n_components = 100
arr = 10 * (np.random.rand(n_components, n_features) * 2 - 1)


def prox_l1_numpy(P, strength):
    P = np.sign(P) * np.maximum(np.abs(P) - strength, 0)
    return P


def prox_l21_numpy(P, strength):
    # P.shape = (n_components, n_features)
    norm = np.sqrt(np.sum(P**2, axis=0))
    return np.maximum(1 - strength / norm, 0) * P


def prox_squaredl12_numpy(P, strength):
    # P.shape = (n_components, n_features)
    P_abs_sorted = np.sort(np.abs(P), axis=1)[:, ::-1]
    ind = np.arange(P.shape[1]) + 1
    S = np.cumsum(P_abs_sorted, axis=1) / (1 + 2 * strength * ind)
    S_rho = []
    for s in range(P.shape[0]):
        rho = P.shape[1] - 1
        for j in range(P.shape[1]):
            if P_abs_sorted[s, j] < 2 * strength * S[s, j]:
                rho = j - 1
                break
        S_rho.append(S[s, rho])
    result = np.abs(P) - 2 * strength * np.array(S_rho)[:, None]
    return np.sign(P) * np.maximum(result, 0)


def prox_squaredl21_numpy(P, strength):
    # P.shape = (n_components, n_features)
    norms = np.linalg.norm(P, ord=2, axis=0)
    norms_proxed = prox_squaredl12_numpy(norms[None, :], strength)
    result = np.array(P)
    print(norms)
    result[:, norms > 0] /= norms
    result *= norms_proxed
    return result


@pytest.mark.parametrize("strength", [0.0001, 0.001, 0.01, 0.1, 1, 10])
def test_prox_squaredl12_numba(strength):
    arr_prox_slow = prox_squaredl12_numpy(np.array(arr), strength)
    arr_prox = np.array(arr.T)
    regularizer = SquaredL12(transpose=True)

    regularizer.init_cache_psgd(2, n_features, n_components)
    regularizer.prox(arr_prox, strength, 2)
    assert_array_equal(
        np.abs(arr) >= np.abs(arr_prox_slow), np.ones(arr.shape, dtype=bool)
    )
    assert_array_almost_equal(arr_prox.T, arr_prox_slow)


@pytest.mark.parametrize("strength", [0.0001, 0.001, 0.01, 0.1, 1, 10])
def test_prox_squaredl21_numba(strength):
    arr_prox_slow = prox_squaredl21_numpy(np.array(arr), strength)
    arr_prox = np.array(arr.T)
    regularizer = SquaredL21()

    regularizer.init_cache_psgd(2, n_features, n_components)
    regularizer.prox(arr_prox, strength, 2)
    assert_array_equal(
        np.abs(arr) >= np.abs(arr_prox_slow), np.ones(arr.shape, dtype=bool)
    )
    assert_array_almost_equal(arr_prox.T, arr_prox_slow)
