# `reports/` — documentos LaTeX

Fuentes `.tex` del informe principal y de las slides. Las figuras vienen
de `../img/` (committed), `../out/figures/` (regenerable) y
`../report_assets/` (outputs de scripts).

## Inventario

| Archivo | Tipo | Contenido |
|---------|------|-----------|
| `Informe_Kernels_Integrado.tex` | Artículo (article) | Informe metodológico completo: bandwidth, gridsize, comparación de los 6 kernels, recomendación final. |
| `Slides_Cosine_Approx.tex` | Beamer 16:9 | 15 slides sobre las aproximaciones polinómicas del kernel coseno (Taylor, Chebyshev, Remez, Bhaskara). |

## Compilación

Desde la raíz del proyecto:

```bash
cd Kernel_Tests
pdflatex -output-directory=out reports/Informe_Kernels_Integrado.tex
pdflatex -output-directory=out reports/Informe_Kernels_Integrado.tex   # 2ª pasada para refs
```

O usar el orquestador PowerShell (recomendado):

```powershell
.\Kernel_Tests\powershell\build_kernel_report.ps1
```

Que ejecuta `build_kernel_report_assets.py` antes de compilar y deja el
PDF en `../out/Informe_Kernels_Integrado.pdf`.

## Paths de figuras (relativos a `reports/`)

- `../img/cosine_approx_error.png` — committed (Slides)
- `../img/cosine_approx_pareto.png` — committed (Slides)
- `../out/figures/kde_gridsize_sensitivity*.png` — regenerable (Informe)
- `../report_assets/kernel_*.png` — assets del script `build_kernel_report_assets.py` (Informe)

## Notas

- Los archivos auxiliares (`.aux`, `.log`, `.fls`, `.synctex.gz`, etc.)
  están en `.gitignore` y se generan en `../out/` cuando se usa
  `-output-directory=out`.
- Si un includegraphics falla, verificar que la figura existe ejecutando
  `python ../scripts/generate_all_figures.py`.
