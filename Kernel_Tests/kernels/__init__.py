"""Utilidades KDE para la etapa ``Kernel_Tests``.

El paquete conserva una sola fuente de verdad para los seis kernels usados
por el proyecto y para la evaluacion KDE univariada.
"""
from __future__ import annotations

KERNELS = ("gaussian", "epanechnikov", "tophat", "exponential", "linear", "cosine")

COLOR_MAP = {
    "gaussian":     "#4c78a8",
    "epanechnikov": "#e45756",
    "tophat":       "#f58518",
    "exponential":  "#54a24b",
    "linear":       "#b279a2",
    "cosine":       "#9d755d",
}

__all__ = ["KERNELS", "COLOR_MAP"]
