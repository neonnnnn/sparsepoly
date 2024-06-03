# sparsepoly

![black/isort/pytest](https://github.com/neonnnnn/sparsepoly/actions/workflows/python-package.yml/badge.svg)

A sklearn-like library for **sparse factorization machines** and thier variants for classification and regression in Python.

Factorization machines (FMs) are machine learning models based on **feature interaction** (co-occurrence) through polynomial terms.
**Sparse** FMs use not all feature interactions but partial of them by using sparsity-inducing regularization, namely, sparse FMs are FMs with **feature interaction selection**.

This package provides some solvers for optimizing FM-like models with some regularizations.

## Summary of Supported Regularizers and Solvers

### Which Regularizer Should You Use?

| Regularizer \ Purpose | Feature Interaction Selection | Feature Selection |
| ---- | ---- | ---- |
| ```l1```| ✘ | ✔ |
| ```l21``` | ✘ | ✔ |
| ```squaredl12``` (```omegati```) | ✔ | ✔ |
| ```squaredl21``` (```omegacs```) | ✘ | ✔ |

For more detail, please see our paper.

### Which Solver Can You Use?

| | Sparse FMs | Sparse Higher-order FMs | Sparse All-subsets Model |
| ---- | ---- | ---- | ---- |
| ```pcd``` | ```l1```, ```squaredl12```, ```omegati``` | ```l1```, ```omegati```  | ```l1```, ```omegati```|
| ```pbcd``` | ```l1```, ```l21```, ```squaredl21```, ```omegacs``` | ```l1```, ```l21```, ```omegacs```  | ```l21```, ```omegacs``` |
| ```psgd``` | ```l1```, ```l2```, ```squaredl12```, ```squaredl21``` | None | None |

The ```pcd``` and ```pbcd``` algorithms are easy to use and produce a sparse solution, so basically you should use ```pcd``` for feature interaction selection and ```pbcd``` for feature selection.

However, for large-scale datasets, the use of the ```psgd``` is recommended because of its scalability.

## Installation

### pip

```bash
    pip install git+https://github.com/neonnnnn/sparsepoly
```

## References

- Kyohei Atarashi, Satoshi Oyama, and Masahito Kurihara. Factorization Machines with Regularization for Sparse Feature Interactions. Journal of Machine Learning Research, 22(153), pp. 1--50, 2021.

## Authors

-------

- Kyohei Atarashi, 2020-present
