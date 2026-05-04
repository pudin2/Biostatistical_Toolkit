"""Carga de valores OTU positivos para la etapa KDE."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_REL = Path("Datos") / "otu_data_converted.csv"


def _project_root() -> Path:
    """Repo root: dos niveles arriba de este archivo (kernels/ -> Kernel_Tests/ -> root)."""
    return Path(__file__).resolve().parent.parent.parent


def default_data_path() -> Path:
    return _project_root() / _DEFAULT_REL


def load_otu_positives(path: str | Path | None = None, verbose: bool = False) -> np.ndarray:
    """Carga el CSV de OTUs y devuelve un vector 1D de valores finitos > 0."""
    csv_path = Path(path) if path is not None else default_data_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset OTU: {csv_path}")

    df = pd.read_csv(csv_path)
    values = df.select_dtypes(include=[np.number]).to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    positives = values[values > 0]

    if verbose:
        print(f"[data] {csv_path.name}: {len(positives):,} valores positivos "
              f"(de {df.shape[0]} muestras x {df.shape[1]} columnas).")
    return positives


__all__ = ["load_otu_positives", "default_data_path"]
