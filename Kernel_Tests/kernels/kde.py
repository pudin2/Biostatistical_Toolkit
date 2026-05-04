"""Evaluador KDE univariado con backend CPU/GPU opcional.

Uso:

    from kernels.kde import KDEEvaluator
    kde = KDEEvaluator(values, kernel="cosine", bandwidth=4.428)
    pdf = kde.evaluate(x_grid)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import KernelDensity


def _prepare_cuda_windows() -> None:
    """Normaliza CUDA en Windows antes de importar CuPy."""
    if os.name != "nt":
        return

    candidates = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_PATH_V13_1"),
        os.environ.get("CUDA_PATH_V13_0"),
        os.environ.get("CUDA_PATH_V12_9"),
        os.environ.get("CUDA_PATH_V12_8"),
        os.environ.get("CUDA_PATH_V12_6"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if root.name.lower() == "bin":
            root = root.parent
        bin_dir = root / "bin"
        if bin_dir.exists():
            os.environ["CUDA_PATH"] = str(root)
            try:
                os.add_dll_directory(str(bin_dir))
            except (FileNotFoundError, OSError):
                pass
            return


_prepare_cuda_windows()

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

from . import KERNELS
from . import core as _core

DEFAULT_CHUNK = 128


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
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """KDE CPU usando la implementacion optimizada de scikit-learn."""
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(data.reshape(-1, 1))
    return np.exp(kde.score_samples(x_grid.reshape(-1, 1)))


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
    """Evaluador KDE con seleccion de backend y chunking estable."""

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
        if kernel not in KERNELS:
            raise ValueError(f"Kernel desconocido: {kernel}. Valid: {KERNELS}")
        self.data = np.ascontiguousarray(np.asarray(data, dtype=float).ravel())
        self.data = self.data[np.isfinite(self.data)]
        if self.data.size == 0:
            raise ValueError("KDEEvaluator requiere datos finitos.")
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
        return _evaluate_cpu(self.data, x, self.bandwidth, self.kernel, self.chunk)


def evaluate_kde(
    data: np.ndarray, x_grid: np.ndarray, bandwidth: float, kernel: str,
    backend: str = "auto",
) -> np.ndarray:
    """Atajo funcional equivalente a ``KDEEvaluator(...).evaluate(x_grid)``."""
    return KDEEvaluator(data, kernel, bandwidth, backend=backend).evaluate(x_grid)


__all__ = ["KDEEvaluator", "evaluate_kde", "gpu_available", "DEFAULT_CHUNK"]
