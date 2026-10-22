import numpy as np


def clr_transform(X, pseudocount=1.0):
    X = np.asarray(X, dtype=float)

    if np.any(X < 0):
        raise ValueError(
            "La matriz contiene valores negativos; CLR no es apropiada así."
        )

    X_pc = X + pseudocount
    row_sums = X_pc.sum(axis=1, keepdims=True)

    if np.any(row_sums == 0):
        raise ValueError(
            "Hay filas con suma 0 incluso tras aplicar pseudocount."
        )

    X_closed = X_pc / row_sums
    log_X = np.log(X_closed)
    gm_log = log_X.mean(axis=1, keepdims=True)

    return log_X - gm_log
