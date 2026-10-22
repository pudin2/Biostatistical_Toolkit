"""Diálogos de selección de archivos (tkinter). Importar solo en entornos con pantalla."""
from pathlib import Path

from .load import load_dataframes, supported_extensions


def _filetypes():
    exts = supported_extensions()
    types = [("Archivos soportados", " ".join(f"*{e}" for e in exts))]
    types += [(e.lstrip(".").upper(), f"*{e}") for e in exts]
    types.append(("Todos los archivos", "*.*"))
    return types


def _hidden_root():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def select_files(title="Selecciona uno o varios datasets", multiple=True):
    from tkinter import filedialog

    root = _hidden_root()
    try:
        ask = filedialog.askopenfilenames if multiple else filedialog.askopenfilename
        selected = ask(title=title, filetypes=_filetypes())
    finally:
        root.destroy()
    if not selected:
        return []
    return list(selected) if multiple else [selected]


def load_multiple_dataframes(title="Selecciona uno o varios datasets", **read_kwargs):
    file_paths = select_files(title, multiple=True)

    if not file_paths:
        print("No se seleccionó ningún archivo.")
        return {}

    def report_error(path, exc):
        print(f"Error al cargar {path.name}: {exc}")

    dataframes = load_dataframes(file_paths, on_error=report_error, **read_kwargs)
    for name, df in dataframes.items():
        print(f"Cargado: {name} -> shape {df.shape}")
    return dataframes


def load_single_dataframe(title="Selecciona un dataset", **read_kwargs):
    from .load import load_dataframe_from_path

    file_paths = select_files(title, multiple=False)

    if not file_paths:
        print("No se seleccionó ningún archivo.")
        return None

    df = load_dataframe_from_path(file_paths[0], **read_kwargs)
    print(f"Cargado: {Path(file_paths[0]).name} -> shape {df.shape}")
    return df
