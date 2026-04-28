"""
core
====
Implementacion unica de las 6 funciones kernel K(u) usadas en el estudio.

Cada kernel se evalua sobre un array ``u = (x - x_data) / h``. Las formulas
son las mismas que aparecen en los notebooks (KDE_Estadisticas_MultiKernel,
KDE_Gridsize_Sensitivity_MultiKernel) y en sklearn.neighbors.KernelDensity.

El parametro ``xp`` permite trabajar tanto con NumPy (CPU) como con CuPy
(GPU) sin duplicar codigo: se le pasa el modulo, no el array. Ambos
modulos exponen la misma API (``xp.exp``, ``xp.cos``, ``xp.where``, ...).

Convenio de normalizacion: cada kernel integra a 1 en u (no en x).
La normalizacion final por h se aplica fuera, en el evaluador KDE.

Formulas (referencia: tabla de la slide "Formas funcionales de los 6 kernels"):

    Gaussian:     K(u) = (1/sqrt(2*pi)) * exp(-u^2 / 2)              soporte R
    Epanechnikov: K(u) = (3/4) * (1 - u^2)         si |u| <= 1
    Tophat:       K(u) = 1/2                       si |u| <= 1
    Exponential:  K(u) = (1/2) * exp(-|u|)                            soporte R
    Linear:       K(u) = (1 - |u|)                 si |u| <= 1
    Cosine:       K(u) = (pi/4) * cos(pi*u/2)      si |u| <= 1
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Constantes precomputadas
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_PI_OVER_2 = np.pi / 2.0
_PI_OVER_4 = np.pi / 4.0


def kernel_eval(u: Any, name: str, xp: Any = np) -> Any:
    """
    Evalua K(u) para el kernel solicitado. Funciona con NumPy o CuPy.

    Parameters
    ----------
    u : array
        Distancias normalizadas. Mismo dtype del backend (np.ndarray o cp.ndarray).
    name : str
        Nombre del kernel (uno de los 6).
    xp : module
        ``numpy`` o ``cupy``. Por defecto NumPy.

    Returns
    -------
    array del mismo backend con K(u) elemento a elemento.
    """
    if name == "gaussian":
        return _INV_SQRT_2PI * xp.exp(-0.5 * u * u)

    if name == "exponential":
        return 0.5 * xp.exp(-xp.abs(u))

    abs_u = xp.abs(u)
    inside = abs_u <= 1.0

    if name == "epanechnikov":
        return xp.where(inside, 0.75 * (1.0 - u * u), 0.0)

    if name == "tophat":
        return xp.where(inside, 0.5, 0.0)

    if name == "linear":
        return xp.where(inside, 1.0 - abs_u, 0.0)

    if name == "cosine":
        return xp.where(inside, _PI_OVER_4 * xp.cos(_PI_OVER_2 * u), 0.0)

    raise ValueError(
        f"Kernel desconocido: '{name}'. Validos: "
        "gaussian, epanechnikov, tophat, exponential, linear, cosine."
    )


def kernel_support(name: str) -> tuple[float, float]:
    """Devuelve el soporte de K(u). Usa +/- inf para soporte real."""
    if name in ("gaussian", "exponential"):
        return (-np.inf, np.inf)
    if name in ("epanechnikov", "tophat", "linear", "cosine"):
        return (-1.0, 1.0)
    raise ValueError(f"Kernel desconocido: {name}")


__all__ = ["kernel_eval", "kernel_support"]
