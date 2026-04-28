# `scripts/` — pipelines reproducibles

Scripts Python que regeneran todos los outputs (CSVs, JSONs, PNGs) sin
intervención manual. Cada uno es ejecutable de forma independiente y
añade automáticamente `Kernel_Tests/` al `sys.path` para resolver el
package `kernels/`.

## Inventario

| Script | Propósito | Tiempo aprox. | Salida |
|--------|-----------|---------------|--------|
| `generate_all_figures.py` | Orquestador único: regenera las 4 figuras de referencia con GPU/CPU automático. | ~60 s CPU | `../img/*.png` (committed) + `../out/figures/*.png` (200 DPI) |
| `compute_cv_all_kernels.py` | Bandwidth por CV (k-fold log-likelihood) para los 6 kernels + tabla compendio. | 5–10 min | `../report_assets/kernel_cv_all.csv`, `kernel_cv_curves.png`, `kernel_final_compendium.json` |
| `build_kernel_report_assets.py` | Tablas y figuras Scott común + heatmap JS pairwise + métricas active. | ~30 s | 10 archivos en `../report_assets/` |
| `benchmark_cosine_approx.py` | Benchmark detallado de las aproximaciones del coseno: error vs `cos` exacto, tiempos GPU, impacto KS/masa. | ~2 min | `../report_assets/cosine_approx_*` |

## Cómo ejecutarlos

```bash
# Desde la raíz del repo
python Kernel_Tests/scripts/generate_all_figures.py
python Kernel_Tests/scripts/compute_cv_all_kernels.py
python Kernel_Tests/scripts/build_kernel_report_assets.py
python Kernel_Tests/scripts/benchmark_cosine_approx.py
```

Cada script:
- Resuelve `ROOT = Path(__file__).resolve().parent.parent` (= `Kernel_Tests/`).
- Inserta `ROOT` en `sys.path` para `from kernels.X import ...`.
- Escribe outputs relativos a `ROOT` (no a `cwd`), así funciona desde
  cualquier directorio.

## Backend GPU

`generate_all_figures.py` y `benchmark_cosine_approx.py` detectan CuPy
automáticamente vía `kernels.kde.gpu_available()`. Para forzar CPU,
desinstalar CuPy o ejecutar con CUDA inválido. La salida imprime `[GPU]`
o `[CPU]` claramente.
