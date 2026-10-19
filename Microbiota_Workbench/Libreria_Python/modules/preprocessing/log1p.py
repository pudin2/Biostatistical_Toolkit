import numpy as np


def log1p_transform(X, **kwargs):
    if np.any(X < 0):
        raise ValueError("log1p no admite valores negativos.")

    return np.log1p(X)
