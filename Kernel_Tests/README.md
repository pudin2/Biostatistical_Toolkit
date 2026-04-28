# Kernel_Tests — sub-estudio metodológico de KDE

Compara seis kernels KDE (gaussian, epanechnikov, tophat, exponential,
linear, cosine) sobre los valores OTU positivos del dataset principal
(N = 105 420 puntos, 95 % de ceros). Produce CSVs, figuras y un informe
LaTeX para soportar la elección de kernel en
`Caracterizacion_Estadistica.ipynb`.

## Estructura

```
Kernel_Tests/
├── kernels/            módulo Python reutilizable (importado por todo lo demás)
├── notebooks/          análisis exploratorios (2)
├── scripts/            pipelines reproducibles (4)
├── reports/            documentos LaTeX
├── powershell/         orquestadores .ps1
├── img/                figuras committed para LaTeX
├── out/                outputs efímeros (gitignored)
├── report_assets/      CSVs/JSONs/PNGs estables generados por scripts
├── README.md           este archivo
├── OVERVIEW.txt        descripción larga del sub-estudio
├── requirements.txt    dependencias CPU
└── requirements-gpu.txt  + CuPy opcional
```

## Quickstart

```bash
# Una sola línea regenera las 4 figuras de referencia (img/ + out/figures/)
python Kernel_Tests/scripts/generate_all_figures.py
```

Detecta GPU automáticamente (CuPy si está disponible) y produce:
- `img/kde_all_kernels.png` — 6 KDE con bandwidth AMISE-equivalente
- `img/kde_gridsize_convergence.png` — KDE Gaussian a gridsizes 100–5000
- `img/cosine_approx_error.png` — error de las 6 aproximaciones del coseno
- `img/cosine_approx_pareto.png` — Pareto coste vs precisión

## Pipelines completos

```bash
# CV bandwidth por kernel + tabla compendio (~5–10 min)
python Kernel_Tests/scripts/compute_cv_all_kernels.py

# Assets del informe LaTeX (10 archivos en report_assets/, ~30 s)
python Kernel_Tests/scripts/build_kernel_report_assets.py

# Benchmark detallado de las aproximaciones del coseno
python Kernel_Tests/scripts/benchmark_cosine_approx.py
```

PowerShell (Windows):

```powershell
# Re-ejecutar los notebooks vía nbconvert
.\Kernel_Tests\powershell\run_kernel_notebooks.ps1

# Generar assets + compilar Informe_Kernels_Integrado.pdf en out/
.\Kernel_Tests\powershell\build_kernel_report.ps1
```

## Notebooks

| # | Notebook | Propósito |
|---|----------|-----------|
| 01 | `notebooks/01_kde_kernels_comparativa.ipynb` | KDE comparativo de los 6 kernels (sklearn + GPU CuPy), bandwidth CV, KS/CvM/masa |
| 02 | `notebooks/02_kde_gridsize_convergencia.ipynb` | Convergencia del KDE al variar el tamaño del grid (gaussian + multikernel) |

## Módulo `kernels/`

| Submódulo | Contenido |
|-----------|-----------|
| `__init__.py` | `KERNELS`, `COLOR_MAP` |
| `core.py` | `kernel_eval(u, name, xp=np)` — los 6 kernels en NumPy/CuPy |
| `kde.py` | `KDEEvaluator(data, kernel, bandwidth, backend="auto")` — chunking 1755 |
| `bandwidth.py` | `scott_h`, `silverman_h`, `cv_loglik` |
| `stats.py` | `ks_distance`, `cvm_distance`, `jensen_shannon`, `cdf_from_pdf`, `mode_kde` |
| `data.py` | `load_otu_positives()` — resuelve el path al CSV automáticamente |
| `metadata.py` | `GPU_TIMING`, `GRIDSIZE_CONVERGENCE`, `AMISE_PROPS`, `H_EQ_FACTORS` |
| `cosine_approx.py` | Taylor / Chebyshev / Remez / Bhaskara para `cos(πu/2)` |

## Aproximaciones del kernel coseno

| Familia    | Identificador | Coste por evaluación | Error sup medido |
|------------|---------------|----------------------|------------------|
| Taylor     | `taylor4`     | 2 mults (Horner)     | 2.0 · 10⁻²       |
| Taylor     | `taylor6`     | 3 mults (Horner)     | 8.9 · 10⁻⁴       |
| Chebyshev  | `cheb4`       | 2 mults (Horner)     | 1.3 · 10⁻³       |
| Chebyshev  | `cheb6`       | 3 mults (Horner)     | **1.7 · 10⁻⁵**   |
| Minimax    | `remez4`      | 2 mults (Horner)     | 8.5 · 10⁻⁴       |
| Racional   | `bhaskara`    | 1 mult + 1 div       | 1.6 · 10⁻³       |

Errores medidos sobre 200 001 puntos uniformes en `[-1, 1]`.
Recomendación operativa: **Chebyshev grado 6** sustituye `cp.cos` con
diferencia indistinguible en KS y masa positiva.

## Backends

`KDEEvaluator(..., backend="auto")` selecciona:
- **GPU (CuPy)** si hay tarjeta CUDA detectable.
- **CPU (NumPy)** como fallback.

Forzar con `backend="gpu"` o `backend="cpu"`.

## Notas

- `img/` está committeado: los `.tex` lo consumen vía `../img/...`.
- `out/` está gitignored: outputs efímeros, PDFs intermedios, copias 200 DPI.
- `report_assets/` contiene los outputs de `compute_cv_all_kernels.py` y
  `build_kernel_report_assets.py`. Se commitean para que `Informe` compile
  sin re-ejecutar pipelines.
- Constantes (`GPU_TIMING`, etc.) viven en `kernels/metadata.py`. Si se
  cambia hardware, ese archivo es lo único que hay que actualizar.
