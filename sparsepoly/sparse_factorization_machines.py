# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.validation import NotFittedError, check_array

from .base import BaseSparsePoly, SparsePolyClassifierMixin, SparsePolyRegressorMixin
from .cd_linear import _cd_linear_epoch
from .dataset import get_dataset
from .kernels import poly_predict
from .loss import CLASSIFICATION_LOSSES, REGRESSION_LOSSES
from .pbcd import pbcd_epoch
from .pcd import pcd_epoch
from .psgd import psgd_epoch
from .regularizer import REGULARIZATION

LEARNING_RATE = {"constant": 0, "optimal": 1, "pegasos": 2, "invscaling": 3}


class _BaseSparseFactorizationMachine(BaseSparsePoly, metaclass=ABCMeta):
    _REGULARIZERS = REGULARIZATION

    @abstractmethod
    def __init__(
        self,
        degree=2,
        loss="squared",
        n_components=2,
        solver="pcd",
        regularizer="squaredl12",
        alpha=1,
        beta=1,
        gamma=1,
        mean=False,
        tol=1e-6,
        fit_lower="explicit",
        fit_linear=True,
        warm_start=False,
        init_lambdas="ones",
        max_iter=100,
        shuffle=False,
        batch_size="auto",
        eta0=1.0,
        learning_rate="optimal",
        power_t=1.0,
        n_iter_no_change=5,
        verbose=False,
        callback=None,
        n_calls=10,
        random_state=None,
    ):
        self.degree = degree
        self.loss = loss
        self.n_components = n_components
        self.solver = solver
        self.regularizer = regularizer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.tol = tol
        self.fit_lower = fit_lower
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.init_lambdas = init_lambdas
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state

    def _augment(self, X):
        # for factorization machines, we add a dummy column for each order.
        if self.fit_lower == "augment":
            k = 2 if self.fit_linear else 1
            for _ in range(self.degree - k):
                X = add_dummy_feature(X, value=1)
        return X

    def _fit_psgd(self, X, y, regularizer, loss_obj, rng):
        n_samples = X.get_n_samples()
        n_features = X.get_n_features()
        converged = False
        indices_samples = np.arange(n_samples, dtype=np.int32)
        no_improvement_count = 0
        best_loss = np.inf
        if self.batch_size == "auto":
            batch_size = int(n_samples * n_features / X.count_nonzero())
        else:
            batch_size = self.batch_size

        if self.learning_rate not in LEARNING_RATE:
            msg = f"learning_rate {self.learning_rate} is not supported."
            msg += f" Choose from {LEARNING_RATE}."
            raise ValueError(msg)
        learning_rate = LEARNING_RATE[self.learning_rate]

        # copy for fast training
        P = np.array(self.P_.swapaxes(1, 2))  # (n_orders, n_features, n_components)

        # init caches and regularizer
        A = np.zeros((P.shape[0], self.degree + 1, self.n_components))
        dA = np.zeros((self.degree, self.n_components))
        grad_P = np.zeros(P.shape)
        grad_w = np.zeros(n_features)
        regularizer.init_cache_psgd(self.degree, n_features, self.n_components)

        # start optimization
        for epoch in range(self.max_iter):
            if self.shuffle:
                rng.shuffle(indices_samples)

            sum_loss, self.it_ = psgd_epoch(
                X,
                y,
                P,
                self.w_,
                self.lams_,
                self.degree,
                self.alpha,
                self.beta,
                self.gamma,
                regularizer,
                loss_obj,
                A,
                dA,
                grad_P,
                grad_w,
                indices_samples,
                self.fit_linear,
                self.eta0,
                learning_rate,
                self.power_t,
                batch_size,
                self.it_,
            )
            # callback
            if (self.callback is not None) and epoch % self.n_calls == 0:
                if self.callback(self) is not None:
                    break

            # stopping criterion
            sum_loss /= n_samples
            if self.verbose:
                print(f"Epoch {epoch+1} loss {sum_loss}")
            if sum_loss > (best_loss - self.tol):
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if sum_loss < best_loss:
                best_loss = sum_loss
            if no_improvement_count >= self.n_iter_no_change:
                if self.verbose:
                    print(f"Converged at iteration {epoch+1}")
                converged = True
                break
        # substitute
        self.P_[:, :, :] = np.array(P.swapaxes(1, 2))
        return converged, epoch

    def _fit_pcd(self, X, y, y_pred, col_norm_sq, regularizer, loss_obj, rng):
        n_samples = X.get_n_samples()
        n_features = X.get_n_features()
        indices_feature = np.arange(n_features, dtype=np.int32)
        indices_component = np.arange(self.n_components, dtype=np.int32)
        converged = False
        if self.mean:
            alpha = self.alpha * n_samples
            beta = self.beta * n_samples
            gamma = self.gamma * n_samples
        else:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
        # caches
        A = np.zeros((n_samples, self.degree + 1))
        dA = np.zeros(self.degree)
        A[:, 0] = 1.0
        # init regularizer
        regularizer.init_cache_pcd(self.degree, n_features, self.n_components)
        # start optimization
        for it in range(self.max_iter):
            viol = 0
            if self.shuffle:
                rng.shuffle(indices_component)
                rng.shuffle(indices_feature)

            if self.fit_linear:
                viol += _cd_linear_epoch(
                    self.w_, X, y, y_pred, col_norm_sq, alpha, loss_obj, indices_feature
                )

            if self.fit_lower == "explicit":
                for deg in range(2, self.degree):
                    viol += pcd_epoch(
                        self.P_[self.degree - deg],
                        X,
                        y,
                        y_pred,
                        self.lams_,
                        deg,
                        beta,
                        gamma,
                        self.eta0,
                        regularizer,
                        loss_obj,
                        A,
                        dA,
                        indices_component,
                        indices_feature,
                    )

            viol += pcd_epoch(
                self.P_[0],
                X,
                y,
                y_pred,
                self.lams_,
                self.degree,
                beta,
                gamma,
                self.eta0,
                regularizer,
                loss_obj,
                A,
                dA,
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
            alpha = self.alpha * n_samples
            beta = self.beta * n_samples
            gamma = self.gamma * n_samples
        else:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
        # caches
        A = np.zeros((n_samples, self.degree + 1, self.n_components))
        dA = np.zeros((n_samples, self.degree, self.n_components))
        grad = np.zeros(self.n_components)
        inv_step_sizes = np.zeros(self.n_components)
        p_j_old = np.zeros(self.n_components)
        A[:, 0] = 1.0

        # init regularizer
        regularizer.init_cache_pbcd(self.degree, n_features, self.n_components)

        # transpose for fast training
        P = np.array(self.P_.swapaxes(1, 2))  # (n_orders, n_features, n_components)

        for it in range(self.max_iter):
            viol = 0
            if self.shuffle:
                rng.shuffle(indices_feature)

            if self.fit_linear:
                viol += _cd_linear_epoch(
                    self.w_, X, y, y_pred, col_norm_sq, alpha, loss_obj, indices_feature
                )

            if self.fit_lower == "explicit":
                for deg in range(2, self.degree):
                    viol += pbcd_epoch(
                        P[self.degree - deg],
                        X,
                        y,
                        y_pred,
                        self.lams_,
                        deg,
                        beta,
                        gamma,
                        self.eta0,
                        regularizer,
                        loss_obj,
                        A,
                        dA,
                        grad,
                        inv_step_sizes,
                        p_j_old,
                        indices_feature,
                    )

            viol += pbcd_epoch(
                P[0],
                X,
                y,
                y_pred,
                self.lams_,
                self.degree,
                beta,
                gamma,
                self.eta0,
                regularizer,
                loss_obj,
                A,
                dA,
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
        self.P_[:, :, :] = np.array(P.swapaxes(1, 2))
        return converged, it

    def fit(self, X, y):
        """Fit factorization machine to training data.

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
        X = self._augment(X)
        n_features = X.shape[1]  # augmented
        rng = check_random_state(self.random_state)
        loss_obj = self._get_loss(self.loss)
        regularizer = self._get_regularizer(self.regularizer)

        if not (self.warm_start and hasattr(self, "w_")):
            self.w_ = np.zeros(n_features, dtype=np.double)

        if self.fit_lower == "explicit":
            n_orders = self.degree - 1
        else:
            n_orders = 1

        if not (self.warm_start and hasattr(self, "P_")):
            self.P_ = 0.01 * rng.randn(n_orders, self.n_components, n_features)

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

        if np.unique(np.abs(self.lams_)) != np.array([1.0]):
            raise ValueError("Lambdas must be +1 or -1.")

        if self.solver in ["pcd", "pbcd"]:
            dataset = get_dataset(X, order="fortran")
            y_pred = self._get_output(X)
            col_norm_sq = row_norms(X.T, squared=True)
        if self.solver in ["psgd"]:
            dataset = get_dataset(X, "c")
            if not (self.warm_start and hasattr(self, "it_")):
                self.it_ = 1

        if self.solver == "pcd":
            converged, self.n_iter_ = self._fit_pcd(
                dataset, y, y_pred, col_norm_sq, regularizer, loss_obj, rng
            )
        elif self.solver == "pbcd":
            converged, self.n_iter_ = self._fit_pbcd(
                dataset, y, y_pred, col_norm_sq, regularizer, loss_obj, rng
            )
        elif self.solver == "psgd":
            dataset = get_dataset(X, order="c")
            converged, self.n_iter_ = self._fit_psgd(
                dataset, y, regularizer, loss_obj, rng
            )
        else:
            msg = f"Solver {self.solver} is not supported."
            raise ValueError(msg)

        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, X):
        y_pred = poly_predict(
            X, self.P_[0, :, :], self.lams_, kernel="anova", degree=self.degree
        )

        if self.fit_linear:
            y_pred += safe_sparse_dot(X, self.w_)

        if self.fit_lower == "explicit" and self.degree == 3:
            # degree cannot currently be > 3
            y_pred += poly_predict(
                X, self.P_[1, :, :], self.lams_, kernel="anova", degree=2
            )

        return y_pred

    def _predict(self, X):
        if not hasattr(self, "P_"):
            raise NotFittedError("Estimator not fitted.")
        X = check_array(X, accept_sparse="csc", dtype=np.double)
        X = self._augment(X)
        return self._get_output(X)


class SparseFactorizationMachineRegressor(
    _BaseSparseFactorizationMachine, SparsePolyRegressorMixin
):
    """Sparse factorization machine for regression (with squared loss).

    Parameters
    ----------

    degree : int >= 2, default: 2
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model.

    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    solver : str, default: 'pcd'.
        {'pcd'|'pbcd'|'psgd'} is supported.

    regularizer : str, default: 'squaredl12'
        A type of sparsity-inducing regularization for higher-order weights.
        {'squaredl12'|'squaredl21'|'omegati'|omegacs'|'l1'|'l21'} are
        supported.

    alpha : float, default: 1
        Regularization amount for linear term (if ``fit_linear=True``).

    beta : float, default: 1
        Regularization amount for higher-order weights.

    gamma : float, default: 1
        Sparsity-inducing regularization amount for higher-order weights.

    mean : boolean, default: False
        Whether loss is mean or sum.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    fit_lower : {'explicit'|'augment'|None}, default: 'explicit'
        Whether and how to fit lower-order, non-homogeneous terms.

        - 'explicit': fits a separate P directly for each lower order.

        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.
           Adds ``degree - 2`` columns if ``fit_linear`` is true, or
           ``degree - 1`` columns otherwise, to account for the linear term.

        - None: only learns weights for the degree given.  If ``degree == 3``,
          for example, the model will only have weights for third-order
          feature interactions.

    fit_linear : boolean, default: True
        Whether to fit an explicit linear term <w, x> to the model, using
        coordinate descent. If False, the model can still capture linear
        effects if ``fit_lower == 'augment'``.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance if the kernel degree is even.  The default value
        of 'ones' matches the original formulation of factorization machines
        (Rendle, 2010).

        To use custom values for the lambdas, ``warm_start`` may be used.

    max_iter : int, optional, default: 100
        Maximum number of passes over the dataset to perform.

    shuffle : boolean, optional, default: True
        Whether cyclic or random order optimization.

    batch_size: int or 'auto', optional, default: 'auto'
        Number of instances in one mini-batch.
        If `auto`, (n_samples * n_features) / X.count_nonzero() is used.
        This is valid when solver='psgd'.

    eta0 : float, default: 1.0
        Step-size parameter. For 'pcd' and 'pbcd' solver, 0.1 or 1.0 is recommended.

    learning_rate : string, default: 'optimal'
        The learning rate schedule:
        - 'constant': `eta = eta0`
        - 'optimal': `eta = eta0 / pow(1.0 + beta * eta0 * t), power_t)`
        - 'pegasos': `eta = 1.0 / (alpha * t)`
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        This is valid when solver='psgd'.

    power_t : float, default: 1.0
        The exponent for inverse scaling and optimal learning rate.
        This is valid when solver='psgd'.

    n_iter_no_change : int, default: 5
        Number of iterations with no improvement to wait before early stopping.
        This is valid when solver='psgd'.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    callback : callable, optional, default: None
        Callback function.

    n_calls : int, optional, default: 10
        Frequency with which `callback` must be called.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.P_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.

        ``self.P_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.

        ``self.P_[i, :, :]`` for i > 0 corresponds to interactions of order
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.w_ : array, shape [n_features]
        The learned linear model, completing the FM.

        Only present if ``self.fit_linear`` is true.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Polynomial Networks and Factorization Machines:
    New Insights and Efficient Training Algorithms.
    Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
    In: Proceedings of ICML 2016.
    http://mblondel.org/publications/mblondel-icml2016.pdf

    Factorization machines.
    Steffen Rendle
    In: Proceedings of IEEE 2010.

    Higher-order Factorization Machines.
    Mathieu Blondel, Akinori Fujino, Naonori Ueda, and Masakazu Ishihata.
    In: Proceedings of NeurIPS 2016.

    Jianpeng Xu, Kaixiang Lin, Pang-Ning Tan, and Jiayu Zhou.
    Synergies that Matter: Efficient Interaction Selection via
    Sparse Factorization Machine.
    In: Proceedings of SDM 2016.

    Huan Zhao, Quanming Yao, Jianda Li, Yangqiu Song, and Dik Lun Lee.
    Meta-graph based Recommendation Fusion over Heterogeneous Information Networks.
    In: Proceedings of KDD 2017.

    Zhen Pan, Enhong Chen, Qi Liu, Tong Xu, Haiping Ma, and Hongjie Lin.
    Sparse Factorization Machines for Click-through Rate Prediction.
    In: Proceedings of ICDM 2016.

    Factorization Machines with Regularization for Sparse Feature Interactions.
    Kyohei Atarashi, Satoshi Oyama, and Masahito Kurihara.
    preprint 2020.
    """

    _LOSSES = REGRESSION_LOSSES

    def __init__(
        self,
        degree=2,
        n_components=2,
        solver="pcd",
        regularizer="squaredl12",
        alpha=1,
        beta=1,
        gamma=1,
        mean=False,
        tol=1e-6,
        fit_lower="explicit",
        fit_linear=True,
        warm_start=False,
        init_lambdas="ones",
        max_iter=100,
        shuffle=False,
        batch_size="auto",
        eta0=1.0,
        learning_rate="optimal",
        power_t=1.0,
        n_iter_no_change=5,
        verbose=False,
        callback=None,
        n_calls=10,
        random_state=None,
    ):
        super(SparseFactorizationMachineRegressor, self).__init__(
            degree,
            "squared",
            n_components,
            solver,
            regularizer,
            alpha,
            beta,
            gamma,
            mean,
            tol,
            fit_lower,
            fit_linear,
            warm_start,
            init_lambdas,
            max_iter,
            shuffle,
            batch_size,
            eta0,
            learning_rate,
            power_t,
            n_iter_no_change,
            verbose,
            callback,
            n_calls,
            random_state,
        )


class SparseFactorizationMachineClassifier(
    _BaseSparseFactorizationMachine, SparsePolyClassifierMixin
):
    """Sparse factorization machine for classification.

    Parameters
    ----------

    degree : int >= 2, default: 2
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model.

    loss : {'logistic'|'squared_hinge'|'squared'}, default: 'squared_hinge'
        Which loss function to use.

        - logistic: L(y, p) = log(1 + exp(-yp))

        - squared hinge: L(y, p) = max(1 - yp, 0)²

        - squared: L(y, p) = 0.5 * (y - p)²

    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    solver: str, default: 'pcd'.
        {'pcd'|'pbcd'|'psgd'} is supported.

    regularizer: str, default: 'squaredl12'
        A type of sparsity-inducing regularization for higher-order weights.
        {'squaredl12'|'squaredl21'|'omegati'|omegacs'|'l1'|'l21'} are
        supported.

    alpha : float, default: 1
        Regularization amount for linear term (if ``fit_linear=True``).

    beta : float, default: 1
        Regularization amount for higher-order weights.

    gamma : float, default: 1
        Sparsity-inducing regularization amount for higher-order weights.

    mean : boolean, default: False
        Whether loss is mean or sum.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    fit_lower : {'explicit'|'augment'|None}, default: 'explicit'
        Whether and how to fit lower-order, non-homogeneous terms.

        - 'explicit': fits a separate P directly for each lower order.

        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.
           Adds ``degree - 2`` columns if ``fit_linear`` is true, or
           ``degree - 1`` columns otherwise, to account for the linear term.

        - None: only learns weights for the degree given.  If ``degree == 3``,
          for example, the model will only have weights for third-order
          feature interactions.

    fit_linear : {True|False}, default: True
        Whether to fit an explicit linear term <w, x> to the model, using
        coordinate descent. If False, the model can still capture linear
        effects if ``fit_lower == 'augment'``.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance if the kernel degree is even.  The default value
        of 'ones' matches the original formulation of factorization machines
        (Rendle, 2010).

        To use custom values for the lambdas, ``warm_start`` may be used.

    max_iter : int, optional, default: 100
        Maximum number of passes over the dataset to perform.

    shuffle : bool, optional, default: True
        Whether cyclic or random order optimization.

    batch_size: int or 'auto', optional, default: 'auto'
        Number of instances in one mini-batch.
        If `auto`, (n_samples * n_features) / X.count_nonzero() is used.
        This is valid when solver='psgd'.

    eta0 : float, default: 1.0
        Step-size parameter. For 'pcd' and 'pbcd' solver, 0.1 or 1.0 is recommended.

    learning_rate : string, default: 'optimal'
        The learning rate schedule:
        - 'constant': `eta = eta0`
        - 'optimal': `eta = eta0 / pow(1.0 + beta * eta0 * t), power_t)`
        - 'pegasos': `eta = 1.0 / (alpha * t)`
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        This is valid when solver='psgd'.

    power_t : float, default: 1.0
        The exponent for inverse scaling and optimal learning rate.
        This is valid when solver='psgd'.

    n_iter_no_change : int, default: 5
        Number of iterations with no improvement to wait before early stopping.
        This is valid when solver='psgd'.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    callback : callable, optional, default: None
        Callback function.

    n_calls : int, optional, default: 10
        Frequency with which `callback` must be called.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.P_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.

        ``self.P_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.

        ``self.P_[i, :, :]`` for i > 0 corresponds to interactions of order
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.w_ : array, shape [n_features]
        The learned linear model, completing the FM.

        Only present if ``self.fit_linear`` is true.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Polynomial Networks and Factorization Machines:
    New Insights and Efficient Training Algorithms.
    Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
    In: Proceedings of ICML 2016.
    http://mblondel.org/publications/mblondel-icml2016.pdf

    Factorization machines.
    Steffen Rendle
    In: Proceedings of IEEE 2010.

    Higher-order Factorization Machines.
    Mathieu Blondel, Akinori Fujino, Naonori Ueda, and Masakazu Ishihata.
    In: Proceedings of NeurIPS 2016.

    Jianpeng Xu, Kaixiang Lin, Pang-Ning Tan, and Jiayu Zhou.
    Synergies that Matter: Efficient Interaction Selection via
    Sparse Factorization Machine.
    In: Proceedings of SDM 2016.

    Huan Zhao, Quanming Yao, Jianda Li, Yangqiu Song, and Dik Lun Lee.
    Meta-graph based Recommendation Fusion over Heterogeneous Information Networks.
    In: Proceedings of KDD 2017.

    Zhen Pan, Enhong Chen, Qi Liu, Tong Xu, Haiping Ma, and Hongjie Lin.
    Sparse Factorization Machines for Click-through Rate Prediction.
    In: Proceedings of ICDM 2016.

    Factorization Machines with Regularization for Sparse Feature Interactions.
    Kyohei Atarashi, Satoshi Oyama, and Masahito Kurihara.
    preprint 2020.
    """

    _LOSSES = CLASSIFICATION_LOSSES

    def __init__(
        self,
        degree=2,
        loss="squared_hinge",
        n_components=2,
        solver="pcd",
        regularizer="squaredl12",
        alpha=1,
        beta=1,
        gamma=1,
        mean=False,
        tol=1e-6,
        fit_lower="explicit",
        fit_linear=True,
        warm_start=False,
        init_lambdas="ones",
        max_iter=100,
        shuffle=False,
        batch_size="auto",
        eta0=1.0,
        learning_rate="optimal",
        power_t=1.0,
        n_iter_no_change=5,
        verbose=False,
        callback=None,
        n_calls=10,
        random_state=None,
    ):
        super(SparseFactorizationMachineClassifier, self).__init__(
            degree,
            loss,
            n_components,
            solver,
            regularizer,
            alpha,
            beta,
            gamma,
            mean,
            tol,
            fit_lower,
            fit_linear,
            warm_start,
            init_lambdas,
            max_iter,
            shuffle,
            batch_size,
            eta0,
            learning_rate,
            power_t,
            n_iter_no_change,
            verbose,
            callback,
            n_calls,
            random_state,
        )
