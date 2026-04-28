# `powershell/` — orquestadores Windows

Scripts `.ps1` que automatizan flujos completos. Asumen Windows con
PowerShell 5+ y la instalación local de Python y MiKTeX.

## Inventario

| Script | Propósito |
|--------|-----------|
| `run_kernel_notebooks.ps1` | Re-ejecuta los 2 notebooks de `../notebooks/` vía nbconvert (in-place, timeout ilimitado). |
| `build_kernel_report.ps1` | Ejecuta `../scripts/build_kernel_report_assets.py` y compila `../reports/Informe_Kernels_Integrado.tex` a `../out/Informe_Kernels_Integrado.pdf` (2 pasadas). |

## Cómo ejecutarlos

```powershell
# Desde cualquier directorio
.\Kernel_Tests\powershell\run_kernel_notebooks.ps1
.\Kernel_Tests\powershell\build_kernel_report.ps1
```

Ambos hacen `Set-Location` internamente al parent (`Kernel_Tests/`), así
que pueden invocarse desde cualquier cwd.

`run_kernel_notebooks.ps1` acepta un parámetro opcional para ejecutar un
subconjunto:

```powershell
.\powershell\run_kernel_notebooks.ps1 -Notebooks "notebooks/01_kde_kernels_comparativa.ipynb"
```

## Dependencias externas

- **Python** con los paquetes de `../requirements.txt` (notebooks).
- **pdflatex** disponible en `PATH` o en el path por defecto de MiKTeX
  (`C:\Users\manue\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`).
  Si tu instalación está en otro lugar, edita `build_kernel_report.ps1`
  línea 8.
