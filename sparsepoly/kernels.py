# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from numba import njit
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse

from .dataset import get_dataset


def safe_power(X, degree=2):
    """Element-wise power supporting both sparse and dense data.

    Parameters
    ----------
    X : ndarray or sparse
        The array whose entries to raise to the power.

    degree : int, default: 2
        The power to which to raise the elements.

    Returns
    -------

    X_ret : ndarray or sparse
        Same shape as X, but (x_ret)_ij = (x)_ij ^ degree
    """
    if issparse(X):
        if hasattr(X, 'power'):
            return X.power(degree)
        else:
            # old scipy
            X = X.copy()
            X.data **= degree
            return X
    else:
        return X ** degree


def _D(X, P, degree=2):
    """The "replacement" part of the homogeneous polynomial kernel.

    D[i, j] = sum_k [(X_ik * P_jk) ** degree]
    """
    return safe_sparse_dot(safe_power(X, degree), P.T ** degree)


def homogeneous_kernel(X, P, degree=2):
    """Convenience alias for homogeneous polynomial kernel between X and P::

        K_P(x, p) = <x, p> ^ degree

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 2

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    return polynomial_kernel(X, P, degree=degree, gamma=1, coef0=0)


def anova_kernel(X, P, degree=2):
    """ANOVA kernel between X and P::

        K_A(x, p) = sum_i1>i2>...>id x_i1 p_i1 x_i2 p_i2 ... x_id p_id

    See John Shawe-Taylor and Nello Cristianini,
    Kernel Methods for Pattern Analysis section 9.2.

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 2

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    if degree == 2:
        K = homogeneous_kernel(X, P, degree=2)
        K -= _D(X, P, degree=2)
        K /= 2
    elif degree == 3:
        K = homogeneous_kernel(X, P, degree=3)
        K -= 3 * _D(X, P, degree=2) * _D(X, P, degree=1)
        K += 2 * _D(X, P, degree=3)
        K /= 6
    else:
        n1 = X.shape[0]
        n2 = P.shape[0]
        Ds = [safe_sparse_dot(X, P.T, dense_output=True)]
        Ds += [_D(X, P, t) for t in range(2, degree+1)]
        anovas = [1., Ds[0]]
        for m in range(2, degree+1):
            anova = np.zeros((n1, n2))
            sign = 1.
            for t in range(1, m+1):
                anova += sign * anovas[m-t] * Ds[t-1]
                sign *= -1.
            anova /= (1.0*m)
            anovas.append(anova)
        K = anovas[-1]
    return K


@njit
def _all_subsets_fast(output, X, P):
    n_samples_x = X.get_n_samples()
    output[:, :] = 1.0
    for i1 in range(n_samples_x):
        n_nz_x, indices_x, data_x = X.get_row(i1)
        for jj in range(n_nz_x):
            j = indices_x[jj]
            n_nz_p, indices_p, data_p = P.get_column(j)
            for ii2 in range(n_nz_p):
                i2 = indices_p[ii2]
                output[i1, i2] *= (1 + data_x[jj]*data_p[ii2])


def all_subsets_kernel(X, P):
    output = np.ones((X.shape[0], P.shape[0]))
    _all_subsets_fast(output,
                      get_dataset(X, order='c'),
                      get_dataset(P, order='fortran'))
    return output


def poly_predict(X, P, lams, kernel, degree=2):
    if kernel == "anova":
        K = anova_kernel(X, P, degree)
    elif kernel == "poly":
        K = homogeneous_kernel(X, P, degree)
    elif kernel == "all-subsets":
        K = all_subsets_kernel(X, P)
    else:
       raise ValueError(("Unsuppported kernel: {}. Use one "
                          "of {{'anova'|'poly'|'all-subsets'}}").format(kernel))
    return np.dot(K, lams)
