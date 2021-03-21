# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

def _cd_linear_epoch(w, X, y, y_pred, col_norm_sq, alpha, loss,
                     indices_feature):
    sum_viol = 0
    n_samples = len(y)
    for j in indices_feature: 
        update = 0
        # compute gradient with respect to w_j
        for i in range(n_samples):
            update += loss.dloss(y_pred[i], y[i]) * X[i, j]
        update += alpha * w[j]
        # compute second derivative upper bound
        inv_step_size = loss.mu * col_norm_sq[j] + alpha
        update /= inv_step_size

        w[j] -= update
        sum_viol += abs(update)

        # update predictions
        for i in range(n_samples):
            y_pred[i] -= update * X[i, j]

    return sum_viol
