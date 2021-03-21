# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
import scipy.sparse as sp
from numba import int32, float64
from numba.experimental import jitclass
from abc import ABCMeta, abstractmethod


spec_dense = [
    ("n_samples", int32),
    ("n_features", int32),
    ("X", float64[:, :]),
    ("indices", int32[:])
]


@jitclass(spec_dense)
class ContiguousDataset(object):
    def __init__(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.X = X
        self.indices = np.arange(self.n_features, dtype=np.int32)

    def get_n_samples(self):
        return self.n_samples

    def get_n_features(self):
        return self.n_features

    def count_nonzero(self):
        return self.n_features * self.n_samples

    def get_row(self, i):
        return self.n_features, self.indices, self.X[i]


@jitclass(spec_dense)
class FortranDataset(object):
    def __init__(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.X = X
        self.indices = np.arange(self.n_samples, dtype=np.int32)
 
    def get_n_samples(self):
        return self.n_samples

    def get_n_features(self):
        return self.n_features
    
    def count_nonzero(self):
        return self.n_features * self.n_samples

    def get_column(self, j):
        return self.n_samples, self.indices, self.X[:, j]


spec_sparse = [
    ("n_samples", int32),
    ("n_features", int32),
    ("data", float64[:]),
    ("indices", int32[:]),
    ("indptr", int32[:])
]

@jitclass(spec_sparse)
class CSRDataset(object):
    def __init__(self, n_samples, n_features, data, indices, indptr):
        self.n_samples = n_samples
        self.n_features = n_features
        self.data = data
        self.indices =  indices
        self.indptr = indptr
 
    def get_n_samples(self):
        return self.n_samples

    def get_n_features(self):
        return self.n_features
    
    def count_nonzero(self):
        return len(self.data)

    def get_row(self, i):
        start = self.indptr[i]
        end = self.indptr[i+1]
        n_nz = end - start
        return n_nz, self.indices[start:end], self.data[start:end]


@jitclass(spec_sparse)
class CSCDataset(object):
    def __init__(self, n_samples, n_features, data, indices, indptr):
        self.n_samples = n_samples
        self.n_features = n_features
        self.data = data
        self.indices =  indices
        self.indptr = indptr

    def get_n_samples(self):
        return self.n_samples

    def get_n_features(self):
        return self.n_features
    
    def count_nonzero(self):
        return len(self.data)

    def get_column(self, j):
        start = self.indptr[j]
        end = self.indptr[j+1]
        n_nz = end - start
        return n_nz, self.indices[start:end], self.data[start:end]


def get_dataset(X, order="c"):
    if sp.isspmatrix(X):
        if order == "fortran":
            X = X.tocsc()
            ds = CSCDataset(X.shape[0], X.shape[1], 
                            X.data, X.indices, X.indptr)
        else:
            X = X.tocsr()
            ds = CSRDataset(X.shape[0], X.shape[1],
                            X.data, X.indices, X.indptr)
    else:
        if order == "fortran":
            X = np.asfortranarray(X, dtype=np.float64)
            ds = FortranDataset(X)
        else:
            X = np.ascontiguousarray(X, dtype=np.float64)
            ds = ContiguousDataset(X)
    return ds
