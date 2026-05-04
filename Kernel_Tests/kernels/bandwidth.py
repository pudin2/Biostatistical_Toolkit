"""Selectores de bandwidth para KDE univariada."""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity


def scott_h(values: np.ndarray) -> float:
    """Bandwidth absoluto por regla de Scott."""
    values = _prepare_values(values)
    std = float(np.std(values, ddof=1))
    return float(gaussian_kde(values, bw_method="scott").factor * std)


def silverman_h(values: np.ndarray) -> float:
    """Bandwidth absoluto por regla de Silverman."""
    values = _prepare_values(values)
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
    """Selecciona bandwidth por k-fold CV de log-verosimilitud."""
    values = _prepare_values(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    k = min(n_subsample, n)
    folds = min(cv_folds, k)
    if folds < 2:
        raise ValueError("cv_loglik requiere al menos 2 observaciones para CV.")

    idx = rng.choice(n, size=k, replace=False)
    sample = values[idx].reshape(-1, 1)

    bw_scott_abs = scott_h(values)
    bw_min = max(0.5, bw_scott_abs * bw_lo_factor)
    bw_max = bw_scott_abs * bw_hi_factor
    bw_grid = np.geomspace(bw_min, bw_max, n_bw)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
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


def _prepare_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("Se requieren al menos 2 valores finitos.")
    if np.nanstd(arr, ddof=1) <= 0:
        raise ValueError("El bandwidth no se puede estimar con varianza cero.")
    return arr


__all__ = ["scott_h", "silverman_h", "cv_loglik"]
