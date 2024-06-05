# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from numba import njit


@njit
def _cd_linear_epoch(w, X, y, y_pred, col_norm_sq, alpha, loss, indices_feature):
    sum_viol = 0
    for j in indices_feature:
        n_nz, indices, data = X.get_column(j)
        update = 0
        # compute gradient with respect to w_j
        for ii in range(n_nz):
            i = indices[ii]
            val = data[ii]
            update += loss.dloss(y_pred[i], y[i]) * val
        update += alpha * w[j]
        # compute second derivative upper bound
        inv_step_size = loss.mu * col_norm_sq[j] + alpha
        update /= inv_step_size

        w[j] -= update
        sum_viol += abs(update)

        # update predictions
        for ii in range(n_nz):
            i = indices[ii]
            val = data[ii]
            y_pred[i] -= update * val

    return sum_viol
