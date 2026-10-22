"""Carga de tablas a DataFrame, independiente del contenido y sin dependencias de GUI.

Los formatos se resuelven por extensión mediante un registro de lectores
(`register_reader`). Para añadir un formato basta registrar una función
`reader(path, **kwargs) -> DataFrame`.
"""
import csv
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

Reader = Callable[..., pd.DataFrame]

_READERS: Dict[str, Reader] = {}

_CANDIDATE_SEPARATORS = ",\t;|"


def register_reader(extensions, reader: Reader) -> None:
    """Asocia una o varias extensiones (con o sin punto) a una función lectora."""
    if isinstance(extensions, str):
        extensions = [extensions]
    for ext in extensions:
        _READERS[_normalize_ext(ext)] = reader


def supported_extensions():
    return sorted(_READERS)


def _normalize_ext(ext: str) -> str:
    ext = ext.lower().strip()
    return ext if ext.startswith(".") else f".{ext}"


def _sniff_separator(path: Path, encoding: Optional[str] = None) -> str:
    """Detecta el separador mirando el inicio del archivo; tabulador si no hay pistas."""
    with open(path, "r", encoding=encoding or "utf-8-sig", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CANDIDATE_SEPARATORS).delimiter
    except csv.Error:
        return "\t"


def _delimited(sep: Optional[str]) -> Reader:
    """Lector de texto delimitado. sep=None detecta el separador automáticamente."""
    def reader(path: Path, **kwargs) -> pd.DataFrame:
        kwargs = dict(kwargs)
        used_sep = kwargs.pop("sep", None) or sep or _sniff_separator(path, kwargs.get("encoding"))
        return pd.read_csv(path, sep=used_sep, **kwargs)
    return reader


register_reader(".csv", _delimited(","))
register_reader([".tsv", ".otus", ".meta", ".taxonomy"], _delimited("\t"))
register_reader(".txt", _delimited(None))
register_reader([".xlsx", ".xls"], lambda path, **kw: pd.read_excel(path, **kw))
register_reader(".parquet", lambda path, **kw: pd.read_parquet(path, **kw))
register_reader(".json", lambda path, **kw: pd.read_json(path, **kw))


def load_dataframe_from_path(path, format: Optional[str] = None, **read_kwargs) -> pd.DataFrame:
    """Carga un archivo como DataFrame.

    format: fuerza la extensión/lector (p. ej. "csv") ignorando la del archivo.
    read_kwargs: se pasan al lector de pandas (sep, encoding, decimal, sheet_name...).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = _normalize_ext(format) if format else path.suffix.lower()
    reader = _READERS.get(ext)

    if reader is None:
        raise ValueError(
            f"Formato no soportado: {ext or '(sin extensión)'}. "
            f"Formatos disponibles: {', '.join(supported_extensions())}"
        )

    return reader(path, **read_kwargs)


def load_dataframes(paths, on_error: Optional[Callable[[Path, Exception], None]] = None, **read_kwargs):
    """Carga varios archivos; devuelve {nombre_sin_extensión: DataFrame}.

    Los archivos que fallan se omiten y se reportan con `on_error(path, exc)` si se da.
    """
    dataframes = {}
    for file_path in paths:
        path = Path(file_path)
        try:
            dataframes[path.stem] = load_dataframe_from_path(path, **read_kwargs)
        except Exception as exc:
            if on_error:
                on_error(path, exc)
    return dataframes
