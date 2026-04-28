# `kernels/` — módulo Python reutilizable

Centraliza la lógica que estaba duplicada en notebooks y scripts. Importable
como `from kernels.X import ...` desde cualquier punto del sub-estudio
(scripts, notebooks, benchmarks).

## Submódulos

| Archivo | Contenido |
|---------|-----------|
| `__init__.py` | Tupla `KERNELS` (los 6 nombres) y `COLOR_MAP` (paleta consistente) |
| `core.py` | `kernel_eval(u, name, xp=np)` — fórmulas K(u) para los 6 kernels, vectorizadas en NumPy o CuPy |
| `kde.py` | `KDEEvaluator(data, kernel, bandwidth, backend="auto", chunk=1755)` — selección automática GPU/CPU + chunking |
| `bandwidth.py` | `scott_h`, `silverman_h`, `cv_loglik` (k-fold log-verosimilitud) |
| `stats.py` | `ks_distance`, `cvm_distance`, `jensen_shannon`, `positive_mass`, `mode_kde`, `cdf_from_pdf`, `cdf_interpolator`, `l1_distance`, `ks_between_cdfs` |
| `data.py` | `load_otu_positives(path=None)` — resuelve `Datos/otu_data_converted.csv` automáticamente |
| `metadata.py` | Constantes hardware/AMISE: `GPU_TIMING`, `GRIDSIZE_CONVERGENCE`, `AMISE_PROPS`, `H_EQ_FACTORS` |
| `cosine_approx.py` | Aproximaciones de `cos(πu/2)` en `[-1,1]`: Taylor 4/6, Chebyshev 4/6, Remez 4, Bhaskara I. Diccionario `APPROXIMATIONS` |

## Decisiones de diseño

- **Late binding** en `kde.py`: `from . import core as _core` y luego
  `_core.kernel_eval(...)`. Esto permite que el benchmark de aproximaciones
  monkey-patchee `core.kernel_eval` y la sustitución se vea reflejada en el
  evaluador KDE sin modificarlo.
- **Backend agnóstico**: las funciones de `core.py` reciben `xp = np | cp`
  como parámetro, sin imports duros de CuPy.
- **Hardware-only en metadata**: si se cambia GPU, sólo hay que actualizar
  `metadata.py`; los scripts y notebooks siguen funcionando.

## Compatibilidad

El módulo es **aditivo**: los notebooks y scripts existentes lo importan
para reducir duplicación, pero su lógica matemática es 1:1 idéntica a las
versiones inline originales (mismas semillas, mismas defaults, mismos
algoritmos).
