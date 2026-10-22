from .clr import clr_transform
from .log1p import log1p_transform

TRANSFORMS = {
    "log1p": lambda X, pseudocount=1.0: log1p_transform(X),
    "clr": lambda X, pseudocount=1.0: clr_transform(X, pseudocount=pseudocount),
}


def apply_feature_transform(X, method="none", pseudocount=1.0):
    """Aplica la transformacion `method` ('none', 'log1p', 'clr') a la matriz X."""
    method = str(method).lower()

    if method == "none":
        return X

    if method not in TRANSFORMS:
        raise ValueError(f"Transformación no soportada: {method}")

    return TRANSFORMS[method](X, pseudocount=pseudocount)
