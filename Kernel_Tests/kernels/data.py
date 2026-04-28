"""
data
====
Carga del vector aplanado de valores OTU > 0.

Pensado como fuente unica para que cualquier script o notebook nuevo
no necesite re-implementar el aplanado y el filtro de positivos.

Resolucion de ruta:
    1. Si se pasa ``path`` explicito -> usa ese.
    2. Si no, busca ``Datos/otu_data_converted.csv`` relativo al repo
       (subiendo desde la ubicacion de este archivo).
"""
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
    """
    Carga el CSV de OTUs y devuelve un vector 1D con todos los valores > 0
    (finitos), aplanado en orden C.

    Es exactamente la misma operacion que ``load_positive_values`` en
    ``compute_cv_all_kernels.py`` y ``build_kernel_report_assets.py``,
    extraida aqui para reuso.

    Parameters
    ----------
    path : str | Path | None
        Ruta al CSV. Si es ``None`` usa ``Datos/otu_data_converted.csv``
        relativo al repositorio.
    verbose : bool
        Si ``True`` imprime un resumen breve.

    Returns
    -------
    np.ndarray (1D, float)
    """
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
