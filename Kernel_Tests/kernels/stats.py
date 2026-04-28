"""
stats
=====
Estadisticas usadas para juzgar la calidad de un KDE univariado.

Todas operan sobre una densidad ya evaluada en una grilla x_grid:

    pdf : np.ndarray (no normalizada todavia)
    x_grid : np.ndarray (mismo length, ordenado ascendente)

La normalizacion (densidad condicional en x > 0) se hace internamente
cuando hace falta. Misma convencion que
``build_kernel_report_assets.fit_conditional_density``.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import cramervonmises, kstest


def positive_mass(pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """Integral de pdf sobre x_grid (regla del trapecio)."""
    return float(np.trapezoid(pdf, x_grid))


def normalize_conditional(pdf: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Normaliza pdf para que sea densidad condicional en el dominio del grid.
    Devuelve (pdf_normalizada, masa_original)."""
    mass = positive_mass(pdf, x_grid)
    return pdf / mass, mass


def cdf_from_pdf(pdf: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """CDF acumulada por suma de Riemann (mismo metodo que los scripts)."""
    dx = np.diff(x_grid, prepend=x_grid[0])
    cdf = np.cumsum(pdf * dx)
    cdf[-1] = 1.0
    return cdf


def cdf_interpolator(pdf: np.ndarray, x_grid: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolador lineal de la CDF, util para ks/cvm."""
    cdf = cdf_from_pdf(pdf, x_grid)
    return interp1d(
        x_grid, cdf, kind="linear",
        bounds_error=False, fill_value=(0.0, 1.0),
        assume_sorted=True,
    )


def mode_kde(pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """Punto x donde pdf alcanza su maximo."""
    return float(x_grid[int(np.argmax(pdf))])


def ks_distance(test_sample: np.ndarray, pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """Estadistico KS entre la muestra de test y la CDF empirica del KDE."""
    f = cdf_interpolator(pdf, x_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kstest(test_sample, lambda z, ff=f: np.asarray(ff(z), dtype=float), method="asymp")
    return float(res.statistic)


def cvm_distance(test_sample: np.ndarray, pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """Estadistico Cramer-von Mises entre muestra y CDF KDE."""
    f = cdf_interpolator(pdf, x_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cramervonmises(test_sample, lambda z, ff=f: np.asarray(ff(z), dtype=float))
    return float(res.statistic)


def jensen_shannon(pdf_a: np.ndarray, pdf_b: np.ndarray, x_grid: np.ndarray, eps: float = 1e-15) -> float:
    """
    Divergencia Jensen-Shannon entre dos densidades sobre la misma grilla.
    Misma definicion que ``build_kernel_report_assets.py``.
    """
    mix = 0.5 * (pdf_a + pdf_b)
    js = (
        0.5 * np.trapezoid(pdf_a * np.log((pdf_a + eps) / (mix + eps)), x_grid)
        + 0.5 * np.trapezoid(pdf_b * np.log((pdf_b + eps) / (mix + eps)), x_grid)
    )
    return float(js)


def l1_distance(pdf_a: np.ndarray, pdf_b: np.ndarray, x_grid: np.ndarray) -> float:
    """Norma L1 entre dos densidades sobre la misma grilla."""
    return float(np.trapezoid(np.abs(pdf_a - pdf_b), x_grid))


def ks_between_cdfs(pdf_a: np.ndarray, pdf_b: np.ndarray, x_grid: np.ndarray) -> float:
    """Distancia maxima entre dos CDFs (no entre muestra y CDF)."""
    cdf_a = cdf_from_pdf(pdf_a, x_grid)
    cdf_b = cdf_from_pdf(pdf_b, x_grid)
    return float(np.max(np.abs(cdf_a - cdf_b)))


__all__ = [
    "positive_mass", "normalize_conditional", "cdf_from_pdf", "cdf_interpolator",
    "mode_kde", "ks_distance", "cvm_distance", "jensen_shannon",
    "l1_distance", "ks_between_cdfs",
]
