.. -*- mode: rst -*-

sparsepoly
=========

A polylearn-like library for **sparse factorization machines** and thier variants
for classification and regression in Python.

Factorization machines (FM) are machine learning models based on
**feature interaction** (co-occurrence) through polynomial terms.
**Sparse** factorization machines use not all feature interactions but
partial of them by using sparsity-inducing regularization, namely, sparse 
factorization machines are factorization machines with **feature interaction
selection**.

This package provides some solvers for optimizing FM-like models with some regularizations:

- proximal coordinate descent (PCD) algorithm for fitting higher-order factorization machines with
    - L1, SquaredL12, and OmegaTI regularization,
- proximal block coordinate descent (PBCD) algorithm for fitting higher-order factorization machines with
    - L1, L21, SquaredL21, and OmegaCS regularization,
- mini-batch proximal stochastic gradient descent (PSGD) algorithm for fitting factorization machines with
    - L1, L21, SquaredL12, and SquaredL21 regularization,
- PCD algorithm for fitting all-subsets models with
    - L1 and OmegaTI regularization,
- PBCD algorithm for fitting all-subsets model with
    - L21 and OmegaCS regularization,
- `polylearn <https://github.com/scikit-learn-contrib/polylearn>`_-compatible API.

Installation
------------

1. Download the sources by::

    git clone https://github.com/neonnnnn/sparsepoly.git
 
 
2. Install the dependencies::

    pip install -r requirements.txt


3. Install the polylearn, please see `polylearn <https://github.com/scikit-learn-contrib/polylearn>`_ installation.


4. Build and install sparsepoly::

    cd sparsepoly
    python setup.py build
    python setup.py install

References
---------

- Kyohei Atarashi, Satoshi Oyama, and Masahito Kurihara. Factorization Machines with Regularization for Sparse Feature Interactions. Journal of Machine Learning Research, 22(153), pp. 1--50, 2021.

Authors
-------

- Kyohei Atarashi, 2020-present
