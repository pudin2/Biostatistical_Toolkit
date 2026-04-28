"""
kernels
=======
Modulo reutilizable para el subproyecto Kernel_Tests.

Centraliza la logica que estaba duplicada en notebooks y scripts:
    - core:       formulas K(u) de los 6 kernels (CPU NumPy y GPU CuPy)
    - kde:        evaluador KDE con backend GPU (CuPy) o CPU (sklearn) y chunking
    - bandwidth:  reglas Scott / Silverman y cross-validation log-likelihood
    - stats:      KS, Cramer-von Mises, Jensen-Shannon, masa positiva, modas, CDF
    - data:       carga de los valores OTU positivos
    - metadata:   constantes (AMISE, h_eq, GPU_TIMING, GRIDSIZE_CONVERGENCE)
    - cosine_approx: aproximaciones polinomicas / racionales del kernel coseno

Importacion tipica:

    from kernels.data import load_otu_positives
    from kernels.bandwidth import scott_h, cv_loglik
    from kernels.kde import KDEEvaluator
    from kernels.stats import ks_distance, jensen_shannon

El modulo es ADITIVO: scripts y notebooks existentes no estan obligados a
migrar. Sirve como fuente unica para nuevo codigo (ej. cosine_approx).
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
