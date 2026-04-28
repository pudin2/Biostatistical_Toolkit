# `report_assets/` — outputs de los scripts pesados

CSVs, JSONs y PNGs producidos por `../scripts/compute_cv_all_kernels.py` y
`../scripts/build_kernel_report_assets.py`. Se commitean en git para que
`../reports/Informe_Kernels_Integrado.tex` compile sin tener que re-ejecutar
los pipelines (CV tarda 5–10 min).

## Inventario

| Archivo | Generado por | Contenido |
|---------|--------------|-----------|
| `kernel_cv_all.csv` | `compute_cv_all_kernels.py` | Bandwidth CV, KS, CvM, masa para cada kernel bajo `h_cv` propio |
| `kernel_cv_curves.png` | `compute_cv_all_kernels.py` | 6 sub-plots de log-verosimilitud CV vs bandwidth |
| `kernel_final_compendium.json` | `compute_cv_all_kernels.py` | Tabla compendio completa (bandwidth + métricas + GPU + AMISE) |
| `kernel_common_scott_metrics.csv` | `build_kernel_report_assets.py` | Métricas con Scott común para los 6 kernels |
| `kernel_common_scott_overlay.png` | `build_kernel_report_assets.py` | Overlay de las 6 KDE bajo Scott común |
| `kernel_pairwise_common_scott.csv` | `build_kernel_report_assets.py` | Distancias pairwise (JS, KS, L1) entre kernels |
| `kernel_pairwise_js_heatmap.png` | `build_kernel_report_assets.py` | Heatmap 6×6 de divergencia Jensen-Shannon |
| `kernel_active_metrics.csv` | `build_kernel_report_assets.py` | Métricas de los 2 kernels activos (gaussian Scott + epanechnikov CV) |
| `kernel_active_overlay.png` | `build_kernel_report_assets.py` | Overlay de los 2 kernels activos |
| `kernel_report_summary.json` | `build_kernel_report_assets.py` | Resumen completo con metadatos de hardware y datos |
| `kernel_timing_metrics.csv` | `build_kernel_report_assets.py` | Tabla simplificada de timing GPU para slides |
| `cosine_approx_*` | `benchmark_cosine_approx.py` | Benchmark detallado de las aproximaciones del coseno |

## Cómo regenerarlos

```bash
python Kernel_Tests/scripts/compute_cv_all_kernels.py        # ~5–10 min
python Kernel_Tests/scripts/build_kernel_report_assets.py    # ~30 s
python Kernel_Tests/scripts/benchmark_cosine_approx.py       # ~2 min
```

## ¿Cuándo regenerar?

- Cambia el dataset (`Datos/otu_data_converted.csv`).
- Cambia un coeficiente en `kernels.metadata.AMISE_PROPS` o `H_EQ_FACTORS`.
- Cambia hardware GPU y se actualiza `kernels.metadata.GPU_TIMING`.
- Se modifica la lógica de `KDEEvaluator`, `cv_loglik` o `kernel_eval`.

En el resto de casos los assets actuales son válidos.
