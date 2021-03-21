# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from numba import njit
import numpy as np
from .dataset import get_dataset
from polylearn.kernels import anova_kernel, homogeneous_kernel


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


def _poly_predict(X, P, lams, kernel, degree=2):
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
