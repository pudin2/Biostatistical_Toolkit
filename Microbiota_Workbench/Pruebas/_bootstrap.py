"""Hace visible la libreria (Libreria_Python/modules) para los scripts de esta carpeta.

Importar este modulo antes que cualquier `from modules...`.
"""
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent.parent / "Libreria_Python"

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
