# `img/` — figuras committeadas para LaTeX

PNGs estables que los `.tex` de `../reports/` consumen vía `../img/...`.
Se commitean en git (a diferencia de `../out/figures/`, que es efímero)
para que cualquiera pueda compilar el informe sin tener que re-ejecutar
los scripts pesados.

## Inventario

| Archivo | DPI | Origen | Consumido por |
|---------|-----|--------|---------------|
| `kde_all_kernels.png` | 140 | `scripts/generate_all_figures.py` | (referencia) |
| `kde_gridsize_convergence.png` | 140 | `scripts/generate_all_figures.py` | (referencia) |
| `cosine_approx_error.png` | 140 | `scripts/generate_all_figures.py` | `reports/Slides_Cosine_Approx.tex` |
| `cosine_approx_pareto.png` | 140 | `scripts/generate_all_figures.py` | `reports/Slides_Cosine_Approx.tex` |

## Cómo regenerarlas

```bash
python Kernel_Tests/scripts/generate_all_figures.py
```

Sobreescribe estos 4 PNGs y produce copias 200 DPI en `../out/figures/`.

## ¿Por qué committearlas?

El informe LaTeX debe compilar de forma reproducible aunque CuPy/CUDA no
estén instalados o el dataset no esté disponible. Mantener `img/` en git
desacopla la compilación del PDF de la regeneración de las figuras.
