"""
metadata
========
Constantes hardcodeadas que estaban duplicadas en
``compute_cv_all_kernels.py`` y ``build_kernel_report_assets.py``.

Centralizar evita inconsistencias y facilita actualizaciones cuando
se mide el rendimiento en hardware nuevo.

Las claves se mantienen identicas a las usadas en los scripts originales
para que la migracion sea un simple ``from kernels.metadata import ...``.
"""
from __future__ import annotations

# Tiempos GPU medidos en KDE_Gridsize_Sensitivity_MultiKernel.ipynb
# Hardware: NVIDIA GPU (8.6 GB VRAM), CUDA 13.1, CuPy
# N=105,420 puntos de datos, gridsize_ref=100,000, chunk ~1755 puntos.
GPU_TIMING: dict[str, dict[str, float]] = {
    "gaussian":     {"t_ref_s": 14.52, "t_sens_s": 713.9},
    "epanechnikov": {"t_ref_s": 15.23, "t_sens_s": 748.0},
    "tophat":       {"t_ref_s":  3.98, "t_sens_s": 195.1},
    "exponential":  {"t_ref_s":  6.54, "t_sens_s": 321.2},
    "linear":       {"t_ref_s":  6.46, "t_sens_s": 317.6},
    "cosine":       {"t_ref_s":  8.40, "t_sens_s": 335.0},
}

# Convergencia con respecto a gridsize bajo Scott h=88.57.
# delta_opt = max |kde(gridsize=10k) - kde(gridsize=100k)|.
GRIDSIZE_CONVERGENCE: dict[str, dict[str, float | bool | int]] = {
    "gaussian":     {"gridsize_opt": 10_000, "delta_opt": 0.0,      "converges": True},
    "epanechnikov": {"gridsize_opt": 10_000, "delta_opt": 0.0,      "converges": True},
    "tophat":       {"gridsize_opt": 10_000, "delta_opt": 0.000596, "converges": False},
    "exponential":  {"gridsize_opt": 10_000, "delta_opt": 0.0,      "converges": True},
    "linear":       {"gridsize_opt": 10_000, "delta_opt": 0.0,      "converges": True},
    "cosine":       {"gridsize_opt": 10_000, "delta_opt": 0.0,      "converges": True},
}

# Eficiencia AMISE relativa (%), factor de escala h_eq vs Gaussian, soporte,
# clase de suavidad y tipo de operacion GPU dominante.
# Los strings con simbolos unicode se preservan tal cual para que coincidan
# byte a byte con los JSON producidos por los scripts originales.
AMISE_PROPS: dict[str, dict[str, float | str]] = {
    "gaussian":     {"amise_pct":  95.1, "h_eq": 1.000, "support": "R",      "ck": "C∞",     "ops": "exp(·)"},
    "epanechnikov": {"amise_pct": 100.0, "h_eq": 2.214, "support": "[-1,1]", "ck": "C⁰",     "ops": "aritmét."},
    "tophat":       {"amise_pct":  92.9, "h_eq": 1.740, "support": "[-1,1]", "ck": "discon.", "ops": "comparac."},
    "exponential":  {"amise_pct":  94.0, "h_eq": 0.740, "support": "R",      "ck": "C⁰",     "ops": "exp(|·|)"},
    "linear":       {"amise_pct":  98.6, "h_eq": 2.432, "support": "[-1,1]", "ck": "C⁰",     "ops": "aritmét."},
    "cosine":       {"amise_pct":  99.9, "h_eq": 2.275, "support": "[-1,1]", "ck": "C¹",     "ops": "cos(·)"},
}

# Factores h_eq usados en build_kernel_report_assets.py para escalar Scott
# cuando se quiere AMISE comparable. Misma fuente que AMISE_PROPS["h_eq"]
# pero con mas decimales (los del script original).
H_EQ_FACTORS: dict[str, float] = {
    "gaussian":     1.00000,
    "epanechnikov": 2.21380,
    "tophat":       1.74006,
    "exponential":  0.73977,
    "linear":       2.43214,
    "cosine":       2.27498,
}

__all__ = ["GPU_TIMING", "GRIDSIZE_CONVERGENCE", "AMISE_PROPS", "H_EQ_FACTORS"]
