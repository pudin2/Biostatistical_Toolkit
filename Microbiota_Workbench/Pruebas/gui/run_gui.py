import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # carpeta Pruebas
import _bootstrap  # noqa: E402,F401  (agrega Libreria_Python al sys.path)

from gui_app import main


if __name__ == "__main__":
    main()
