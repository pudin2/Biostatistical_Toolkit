"""Metricas basicas para densidades KDE evaluadas en una grilla."""
from __future__ import annotations

import numpy as np


def positive_mass(pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """Integral de pdf sobre x_grid (regla del trapecio)."""
    return float(np.trapezoid(pdf, x_grid))


def normalize_conditional(pdf: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Normaliza pdf para que sea densidad condicional en el dominio del grid.
    Devuelve (pdf_normalizada, masa_original)."""
    mass = positive_mass(pdf, x_grid)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("La densidad KDE tiene masa no positiva o no finita.")
    return pdf / mass, mass


def mode_kde(pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """Punto x donde pdf alcanza su maximo."""
    return float(x_grid[int(np.argmax(pdf))])


__all__ = ["positive_mass", "normalize_conditional", "mode_kde"]
