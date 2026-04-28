# `notebooks/` — análisis exploratorios

Notebooks Jupyter con el análisis interactivo. Cada uno carga datos de
`../../Datos/` y produce figuras matplotlib embebidas que el informe LaTeX
referencia o que se exportan a `../out/figures/`.

## Inventario

| Notebook | Propósito |
|----------|-----------|
| `01_kde_kernels_comparativa.ipynb` | KDE de los 6 kernels (gaussian, epanechnikov, tophat, exponential, linear, cosine) sobre los OTU positivos. Incluye bandwidth por CV, KS/CvM/masa, comparación scipy ↔ sklearn, GPU CuPy opcional. |
| `02_kde_gridsize_convergencia.ipynb` | Convergencia del KDE al variar el tamaño del grid (100 → 100 000) para gaussian y multikernel. Genera `../out/figures/kde_gridsize_sensitivity*.png`. |

## Cómo ejecutarlos

```bash
jupyter lab Kernel_Tests/notebooks/
```

O por lote:

```powershell
.\Kernel_Tests\powershell\run_kernel_notebooks.ps1
```

(re-ejecuta los 2 notebooks vía nbconvert, in-place).

## Resolución de paths

Los notebooks usan `Path.cwd().parent.parent / "Datos"` para localizar
`otu_data_converted.csv`. Eso asume que el cwd al ejecutar el notebook es
la propia carpeta del notebook (Jupyter lo hace por defecto al abrir uno).

Para usar el módulo `kernels/` desde un notebook, añadir al primer cell:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # Kernel_Tests/ entra al sys.path
from kernels.kde import KDEEvaluator
```
