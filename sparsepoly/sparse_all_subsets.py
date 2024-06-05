# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array

try:
    from sklearn.exceptions import NotFittedError
except ImportError:

    class NotFittedError(ValueError, AttributeError):
        pass


from sparsepoly.base import (
    BaseSparsePoly,
    SparsePolyClassifierMixin,
    SparsePolyRegressorMixin,
)
from sparsepoly.dataset import get_dataset
from sparsepoly.kernels import poly_predict
from sparsepoly.loss import CLASSIFICATION_LOSSES, REGRESSION_LOSSES
from sparsepoly.optimizer import pbcd_all, pcd_all
from sparsepoly.regularizer import L1, L21, OmegaCS, OmegaTI


class _BaseSparseAllSubsets(BaseSparsePoly, metaclass=ABCMeta):
    _REGULARIZERS = {
        "l1": L1,
        "l21": L21,
        "omegacs": OmegaCS,
        "omegati": OmegaTI,
    }

    @abstractmethod
    def __init__(
        self,
        loss="squared",
        n_components=2,
        solver="pcd",
        beta=1,
        gamma=1,
        eta0=0.1,
        mean=False,
        tol=1e-6,
        regularizer="omegati",
        warm_start=False,
        init_lambdas="ones",
        max_iter=100,
        shuffle=False,
        verbose=False,
        callback=None,
        n_calls=10,
        random_state=None,
    ):
        self.loss = loss
        self.n_components = n_components
        self.solver = solver
        self.beta = beta
        self.gamma = gamma
        self.eta0 = eta0
        self.mean = mean
        self.tol = tol
        self.regularizer = regularizer
        self.warm_start = warm_start
        self.init_lambdas = init_lambdas
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.verbose = verbose
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state

    def _fit_pcd(self, X, y, y_pred, col_norm_sq, regularizer, loss_obj, rng):
        n_samples = X.get_n_samples()
        n_features = X.get_n_features()
        indices_feature = np.arange(n_features, dtype=np.int32)
        indices_component = np.arange(self.n_components, dtype=np.int32)
        converged = False
        if self.mean:
            beta = n_samples * self.beta
            gamma = n_samples * self.gamma
        else:
            beta = self.beta
            gamma = self.gamma

        # caches
        A = np.ones(n_samples)

        # init regularizer
        regularizer.init_cache_pcd(-1, n_features, self.n_components)

        # start optimization
        it = 0
        for it in range(self.max_iter):
            viol = 0
            if self.shuffle:
                rng.shuffle(indices_component)
                rng.shuffle(indices_feature)

            viol += pcd_all.pcd_epoch(
                self.P_,
                X,
                y,
                y_pred,
                self.lams_,
                beta,
                gamma,
                self.eta0,
                regularizer,
                loss_obj,
                A,
                indices_component,
                indices_feature,
            )

            if (self.callback is not None) and it % self.n_calls == 0:
                if self.callback(self) is not None:
                    break

            if self.verbose:
                print(f"Iteration {it+1} violation sum {viol}")

            if viol < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {it+1}")
                converged = True
                break
        return converged, it

    def _fit_pbcd(self, X, y, y_pred, col_norm_sq, regularizer, loss_obj, rng):
        n_samples = X.get_n_samples()
        n_features = X.get_n_features()
        converged = False
        indices_feature = np.arange(n_features, dtype=np.int32)
        if self.mean:
            beta = n_samples * self.beta
            gamma = n_samples * self.gamma
        else:
            beta = self.beta
            gamma = self.gamma

        # caches
        A = np.ones((n_samples, self.n_components))
        grad = np.zeros(self.n_components)
        inv_step_sizes = np.zeros(self.n_components)
        p_j_old = np.zeros(self.n_components)

        # init regularizer
        regularizer.init_cache_pbcd(-1, n_features, self.n_components)

        # transpose for fast training
        P = np.array(self.P_.T)  # (n_orders, n_features, n_components)

        # start optimization
        it = 0
        for it in range(self.max_iter):
            viol = 0
            if self.shuffle:
                rng.shuffle(indices_feature)

            viol += pbcd_all.pbcd_epoch(
                P,
                X,
                y,
                y_pred,
                self.lams_,
                beta,
                gamma,
                self.eta0,
                regularizer,
                loss_obj,
                A,
                grad,
                inv_step_sizes,
                p_j_old,
                indices_feature,
            )

            if (self.callback is not None) and it % self.n_calls == 0:
                if self.callback(self) is not None:
                    break

            if self.verbose:
                print(f"Iteration {it+1} violation sum {viol}")

            if viol < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {it+1}")
                converged = True
                break

        # substitute
        self.P_[:, :] = np.array(P.T)
        return converged, it

    def fit(self, X, y):
        """Fit all-subsets model to training data.
        Parameters
        ----------
        X : array-like or sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : Estimator
            Returns self.
        """

        X, y = self._check_X_y(X, y)
        n_features = X.shape[1]
        dataset = get_dataset(X, order="fortran")
        rng = check_random_state(self.random_state)
        loss_obj = self._get_loss(self.loss)
        regularizer = self._get_regularizer(self.regularizer)
        if not (self.warm_start and hasattr(self, "P_")):
            self.P_ = 0.01 * rng.randn(self.n_components, n_features)

        if not (self.warm_start and hasattr(self, "lams_")):
            if self.init_lambdas == "ones":
                self.lams_ = np.ones(self.n_components)
            elif self.init_lambdas == "random_signs":
                self.lams_ = np.sign(rng.randn(self.n_components))
            else:
                raise ValueError(
                    "Lambdas must be initialized as ones "
                    "(init_lambdas='ones') or as random "
                    "+/- 1 (init_lambdas='random_signs')."
                )

        dataset = get_dataset(X, order="fortran")
        y_pred = self._get_output(X)
        col_norm_sq = row_norms(X.T, squared=True)

        if self.solver == "pcd":
            converged, self.n_iter_ = self._fit_pcd(
                dataset, y, y_pred, col_norm_sq, regularizer, loss_obj, rng
            )
        elif self.solver == "pbcd":
            converged, self.n_iter_ = self._fit_pbcd(
                dataset, y, y_pred, col_norm_sq, regularizer, loss_obj, rng
            )
        else:
            msg = f"Solver {self.solver} is not supported."
            raise ValueError(msg)

        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, X):
        y_pred = poly_predict(X, self.P_, self.lams_, "all-subsets")

        return y_pred

    def _predict(self, X):
        if not hasattr(self, "P_"):
            raise NotFittedError("Estimator not fitted.")
        X = check_array(X, accept_sparse="csc", dtype=np.double)
        return self._get_output(X)


class SparseAllSubsetsRegressor(_BaseSparseAllSubsets, SparsePolyRegressorMixin):
    """Sparse All-subsets model for regression (with squared loss).
    Parameters
    ----------
    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    solver: str, defualt: 'pcd'
        {'pcd'|'pbcd'} is supported.

    beta : float, default: 1
        Regularization amount for interaction weights.

    gamma : float, default: 1
        Sparsity-inducing regularization amount for interaction weights.

    mean : boolean, default: False
        Whether loss is mean or sum.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    regularizer : str, default: 'omegati'
        A type of sparsity-inducing regularization for interaction weights.
        {'omegati'|omegacs'|'l1'|'l21'} are supported now.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance. The default value of 'ones' matches the original
        formulation.
        Using custom lambdas is not supported.
        Only 1 and -1 are allowed for lams[s].

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    shuffle : bool, optional, default: False
        Whether cyclic or random order optimization.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------
    self.P_ : array, shape [n_components, n_features]
        The learned basis functions.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Higher-Order Factorization Machines.
    Mathieu Blondel, Akinori Fujino, Naonori Ueda, Masakazu Ishihata.
    In: Proceedings of NIPS 2016.

    Factorization Machines with Regularization for Sparse Feature Interactions.
    Kyohei Atarashi, Satoshi Oyama, and Masahito Kurihara.
    preprint 2020.

    """

    _LOSSES = REGRESSION_LOSSES

    def __init__(
        self,
        n_components=2,
        solver="pcd",
        beta=1,
        gamma=1,
        eta0=0.1,
        mean=False,
        tol=1e-6,
        regularizer="omegati",
        warm_start=False,
        init_lambdas="ones",
        max_iter=100,
        shuffle=False,
        verbose=False,
        callback=None,
        n_calls=10,
        random_state=None,
    ):
        super(SparseAllSubsetsRegressor, self).__init__(
            "squared",
            n_components,
            solver,
            beta,
            gamma,
            eta0,
            mean,
            tol,
            regularizer,
            warm_start,
            init_lambdas,
            max_iter,
            shuffle,
            verbose,
            callback,
            n_calls,
            random_state,
        )


class SparseAllSubsetsClassifier(_BaseSparseAllSubsets, SparsePolyClassifierMixin):
    """Sparse all-subsets model for classification.
    Parameters
    ----------
    loss : {'logistic'|'squared_hinge'|'squared'}, default: 'squared_hinge'
        Which loss function to use.
        - logistic: L(y, p) = log(1 + exp(-yp))
        - squared hinge: L(y, p) = max(1 - yp, 0)²
        - squared: L(y, p) = 0.5 * (y - p)²

    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    solver: str, defualt: 'pcd'
        {'pcd'|'pbcd'} is supported.

    beta : float, default: 1
        Regularization amount for feature combinations weights.

    gamma : float, default: 1
        Sparsity-inducing regularization amount for interaction weights.

    mean : boolean, default: False
        Whether loss is mean or sum.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    regularizer : str, default='omegati'
        A type of sparsity-inducing regularization for interaction weights.
        {'omegati'|omegacs'|'l1'|'l21'} are supported now.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance. The default value of 'ones' matches the original
        formulation.
        Using custom lambdas is not supported.
        Only 1 and -1 are allowed for lams[s].

    max_iter : int, optional, default: 100
        Maximum number of passes over the dataset to perform.

    shuffle : bool, optional, default: False
        Whether cyclic or random order optimization.

    eta0 : float, default: 1.0
        Step-size parameter. For 'pcd' and 'pbcd' solver,
        0.1 or 1.0 is recommended.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    callback : callable, optinal, default: None
        Callback function.

    n_calls : int, optinal, default: 10
        Frequency with which `callback` must be called.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------
    self.P_ : array, shape [n_components, n_features]
        The learned basis functions.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Higher-Order Factorization Machines.
    Mathieu Blondel, Akinori Fujino, Naonori Ueda, Masakazu Ishihata.
    In: Proceedings of NIPS 2016.

    Factorization Machines with Regularization for Sparse Feature Interactions.
    Kyohei Atarashi, Satoshi Oyama, and Masahito Kurihara.
    preprint 2020.
    """

    _LOSSES = CLASSIFICATION_LOSSES

    def __init__(
        self,
        loss="squared_hinge",
        n_components=2,
        solver="pcd",
        beta=1,
        gamma=1,
        eta0=0.1,
        mean=False,
        regularizer="omegati",
        tol=1e-6,
        warm_start=False,
        init_lambdas="ones",
        max_iter=100,
        shuffle=False,
        verbose=False,
        callback=None,
        n_calls=10,
        random_state=None,
    ):
        super(SparseAllSubsetsClassifier, self).__init__(
            loss,
            n_components,
            solver,
            beta,
            gamma,
            eta0,
            mean,
            tol,
            regularizer,
            warm_start,
            init_lambdas,
            max_iter,
            shuffle,
            verbose,
            callback,
            n_calls,
            random_state,
        )
