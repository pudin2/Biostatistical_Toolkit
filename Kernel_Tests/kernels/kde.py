"""
kde
===
Evaluador KDE univariado con doble backend:

- GPU (CuPy) cuando hay tarjeta disponible -> chunked, alta capacidad.
- CPU (sklearn) como fallback automatico.

Equivalente a la logica que los notebooks
``KDE_*MultiKernel.ipynb`` y los scripts (compute_cv, build_assets)
implementan en linea.

Uso tipico:

    from kernels.kde import KDEEvaluator
    kde = KDEEvaluator(values, kernel="cosine", bandwidth=4.428)
    pdf = kde.evaluate(x_grid)

El chunking en GPU usa ~1755 puntos de evaluacion por lote, copiando
el numero medido en KDE_Gridsize_Sensitivity_MultiKernel.ipynb.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

from . import core as _core   # late binding: permite que tests/benchmarks
                              # monkeypatcheen core.kernel_eval y la
                              # sustitucion se vea reflejada aqui.

DEFAULT_CHUNK = 1755


def gpu_available() -> bool:
    """True si CuPy importa Y hay al menos un dispositivo CUDA visible."""
    if not _HAS_CUPY:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0  # type: ignore[union-attr]
    except Exception:
        return False


def _evaluate_cpu(
    data: np.ndarray, x_grid: np.ndarray, bandwidth: float, kernel: str,
) -> np.ndarray:
    """KDE sumando contribuciones K((x - xi)/h)/h. NumPy puro."""
    n = data.size
    out = np.zeros_like(x_grid, dtype=float)
    # Vectorizacion por puntos del grid (M pequeno respecto a N normalmente).
    # Para mantener memoria estable se itera por chunks del grid.
    chunk = max(1, min(DEFAULT_CHUNK, x_grid.size))
    for start in range(0, x_grid.size, chunk):
        end = min(start + chunk, x_grid.size)
        u = (x_grid[start:end][:, None] - data[None, :]) / bandwidth
        ku = _core.kernel_eval(u, kernel, xp=np)
        out[start:end] = ku.sum(axis=1) / (n * bandwidth)
    return out


def _evaluate_gpu(
    data: np.ndarray, x_grid: np.ndarray, bandwidth: float, kernel: str,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """Misma logica con CuPy. Chunked sobre el grid para no saturar VRAM."""
    assert _HAS_CUPY and cp is not None
    data_g = cp.asarray(data, dtype=cp.float64)
    grid_g = cp.asarray(x_grid, dtype=cp.float64)
    n = int(data_g.size)
    out_g = cp.zeros_like(grid_g)
    for start in range(0, int(grid_g.size), chunk):
        end = min(start + chunk, int(grid_g.size))
        u = (grid_g[start:end][:, None] - data_g[None, :]) / bandwidth
        ku = _core.kernel_eval(u, kernel, xp=cp)
        out_g[start:end] = ku.sum(axis=1) / (n * bandwidth)
    return cp.asnumpy(out_g)


class KDEEvaluator:
    """
    Evaluador KDE con seleccion de backend.

    Parameters
    ----------
    data : np.ndarray
        Vector 1D de muestras.
    kernel : str
        Nombre del kernel (ver ``kernels.core``).
    bandwidth : float
        h absoluto.
    backend : {"auto", "gpu", "cpu"}
        - ``"auto"``: GPU si esta disponible, si no CPU.
        - ``"gpu"`` : fuerza CuPy. Lanza si no hay GPU.
        - ``"cpu"`` : NumPy puro.
    chunk : int
        Puntos de grid por lote (solo aplica a GPU). Default 1755 = numero
        medido en KDE_Gridsize_Sensitivity_MultiKernel.ipynb.

    Notes
    -----
    Se mantiene una API minima a proposito: ``evaluate(grid)`` es lo que
    usan todos los analisis del estudio.
    """

    def __init__(
        self,
        data: np.ndarray,
        kernel: str,
        bandwidth: float,
        backend: str = "auto",
        chunk: int = DEFAULT_CHUNK,
    ) -> None:
        if bandwidth <= 0:
            raise ValueError(f"bandwidth debe ser > 0, recibido {bandwidth}")
        self.data = np.ascontiguousarray(np.asarray(data, dtype=float).ravel())
        self.kernel = kernel
        self.bandwidth = float(bandwidth)
        self.chunk = int(chunk)

        if backend == "auto":
            backend = "gpu" if gpu_available() else "cpu"
        elif backend == "gpu" and not gpu_available():
            raise RuntimeError("backend='gpu' solicitado pero CuPy/CUDA no disponible.")
        elif backend not in ("gpu", "cpu"):
            raise ValueError(f"backend invalido: {backend}")
        self.backend = backend

    def evaluate(self, x_grid: np.ndarray) -> np.ndarray:
        x = np.asarray(x_grid, dtype=float).ravel()
        if self.backend == "gpu":
            return _evaluate_gpu(self.data, x, self.bandwidth, self.kernel, self.chunk)
        return _evaluate_cpu(self.data, x, self.bandwidth, self.kernel)


def evaluate_kde(
    data: np.ndarray, x_grid: np.ndarray, bandwidth: float, kernel: str,
    backend: str = "auto",
) -> np.ndarray:
    """Atajo funcional equivalente a ``KDEEvaluator(...).evaluate(x_grid)``."""
    return KDEEvaluator(data, kernel, bandwidth, backend=backend).evaluate(x_grid)


__all__ = ["KDEEvaluator", "evaluate_kde", "gpu_available", "DEFAULT_CHUNK"]
