# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT
# This code is based on polylearn.base and scikit-learn.

from abc import ABCMeta

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y

from .loss import CLASSIFICATION_LOSSES, REGRESSION_LOSSES


class BaseSparsePoly(BaseEstimator, metaclass=ABCMeta):
    def _get_loss(self, loss):
        if loss not in self._LOSSES:
            losses_str = '", "'.join(self._LOSSES)
            raise ValueError(
                f'Loss function {loss} not supported. The available options are: "{losses_str}".'
            )
        return self._LOSSES[loss]

    def _get_regularizer(self, regularizer):
        if regularizer not in self._REGULARIZERS:
            regularizers_str = '", "'.join(self._REGULARIZERS)
            raise ValueError(
                f'Regularizer {regularizer} not supported. The available options are: "{regularizers_str}".'
            )
        return self._REGULARIZERS[regularizer]()


class SparsePolyRegressorMixin(RegressorMixin):

    _LOSSES = REGRESSION_LOSSES

    def _check_X_y(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=True,
            multi_output=False,
            dtype=np.double,
            y_numeric=True,
        )
        y = y.astype(np.double).ravel()
        return X, y

    def predict(self, X):
        """Predict regression output for the samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Returns predicted values.
        """
        return self._predict(X)


class SparsePolyClassifierMixin(ClassifierMixin):

    _LOSSES = CLASSIFICATION_LOSSES

    def decision_function(self, X):
        """Compute the output of the SparsePoly model before thresholding.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_scores : array, shape = [n_samples]
            Returns predicted values.
        """
        return self._predict(X)

    def predict(self, X):
        """Predict classification output for the samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Returns predicted values.
        """
        y_pred = self.decision_function(X) > 0
        return self.label_binarizer_.inverse_transform(y_pred)

    def predict_proba(self, X):
        """Compute probability estimates for the test samples.

        Only available if `loss='logistic'`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_scores : array, shape = [n_samples]
            Probability estimates that the samples are from the positive class.
        """
        if self.loss == "logistic":
            return 1 / (1 + np.exp(-self.decision_function(X)))
        else:
            raise ValueError(
                "Probability estimates only available for "
                "loss='logistic'. You may use probability "
                "calibration methods from scikit-learn instead."
            )

    def _check_X_y(self, X, y):
        # helpful error message for sklearn < 1.17
        is_2d = hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] >= 2

        if is_2d or type_of_target(y) != "binary":
            raise TypeError(
                "Only binary targets supported. For training "
                "multiclass or multilabel models, you may use the "
                "OneVsRest or OneVsAll metaestimators in "
                "scikit-learn."
            )

        X, Y = check_X_y(X, y, dtype=np.double, accept_sparse=True, multi_output=False)

        self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self.label_binarizer_.fit_transform(Y).ravel().astype(np.double)
        return X, y
