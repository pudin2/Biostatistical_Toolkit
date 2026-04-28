"""
bandwidth
=========
Selectores de ancho de banda h para KDE univariada.

- ``scott_h``       : regla de Scott   h = sigma * n^(-1/5)
- ``silverman_h``   : regla de Silverman, ligeramente robusta a colas
- ``cv_loglik``     : cross-validation por log-verosimilitud (k-fold)

Las dos primeras reproducen el comportamiento de
``scipy.stats.gaussian_kde(...).factor * std`` que es como se calculan
en los scripts originales (compute_cv_all_kernels.py:104,
build_kernel_report_assets.py:90).

``cv_loglik`` envuelve la logica de
``compute_cv_all_kernels.compute_cv_bandwidth`` con los mismos defaults
(submuestra 10000, 5 folds, 40 anchos en geomspace [0.02*Scott, 3*Scott]).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity


def scott_h(values: np.ndarray) -> float:
    """h Scott absoluto. Usa scipy para calcular el factor con los mismos
    convenios que los scripts existentes."""
    std = float(np.std(values, ddof=1))
    return float(gaussian_kde(values, bw_method="scott").factor * std)


def silverman_h(values: np.ndarray) -> float:
    """h Silverman absoluto."""
    std = float(np.std(values, ddof=1))
    return float(gaussian_kde(values, bw_method="silverman").factor * std)


def cv_loglik(
    values: np.ndarray,
    kernel: str,
    cv_folds: int = 5,
    n_subsample: int = 10_000,
    n_bw: int = 40,
    bw_lo_factor: float = 0.02,
    bw_hi_factor: float = 3.0,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Selecciona h por k-fold cross-validation maximizando la log-verosimilitud.

    Reproduce ``compute_cv_all_kernels.compute_cv_bandwidth`` con la misma
    estructura: rango [bw_lo_factor * Scott, bw_hi_factor * Scott] (con piso
    de 0.5 para evitar h = 0).

    Returns
    -------
    h_cv : float
    bw_grid : np.ndarray
    scores : np.ndarray
        Log-verosimilitud media (sobre folds) en cada bw del grid.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    k = min(n_subsample, n)
    idx = rng.choice(n, size=k, replace=False)
    sample = values[idx].reshape(-1, 1)

    bw_scott_abs = scott_h(values)
    bw_min = max(0.5, bw_scott_abs * bw_lo_factor)
    bw_max = bw_scott_abs * bw_hi_factor
    bw_grid = np.geomspace(bw_min, bw_max, n_bw)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = np.zeros(n_bw)
    for i, bw in enumerate(bw_grid):
        fold_scores = []
        for train_idx, val_idx in kf.split(sample):
            kde = KernelDensity(kernel=kernel, bandwidth=bw)
            kde.fit(sample[train_idx])
            fold_scores.append(kde.score(sample[val_idx]))
        scores[i] = float(np.mean(fold_scores))
        if verbose:
            print(f"  [cv {kernel}] bw={bw:.4g}  score={scores[i]:.4g}")

    best_idx = int(np.argmax(scores))
    return float(bw_grid[best_idx]), bw_grid, scores


__all__ = ["scott_h", "silverman_h", "cv_loglik"]
