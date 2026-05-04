# Kernel_Tests

Etapa KDE del proyecto. Esta carpeta estima densidades sobre los valores
positivos de `Datos/otu_data_converted.csv` y genera solo figuras KDE.

## Estructura

```text
Kernel_Tests/
  KDE_Estimaciones.ipynb
  README.md
  requirements.txt
  figures/
  kernels/
    __init__.py
    core.py
    kde.py
    bandwidth.py
    data.py
    stats.py
```

## Flujo

1. Ejecutar `KDE_Estimaciones.ipynb`.
2. El notebook estima tres bandwidths: `cv`, `scott` y `silverman`.
3. Se genera `figures/bandwidth_comparison.png` para comparar las tres opciones.
4. El usuario elige por consola el bandwidth que desea usar. Si presiona Enter,
   se usa `cv`.
5. El notebook sugiere un grid segun el volumen de datos y permite editarlo.
6. Se generan las figuras KDE finales.

## Salidas

- `figures/bandwidth_comparison.png`
- `figures/kde_all_kernels.png`
- `figures/kde_by_kernel_grid.png`

## Kernels disponibles

- `gaussian`
- `epanechnikov`
- `tophat`
- `exponential`
- `linear`
- `cosine`

## Notas

- KDE usa solo valores positivos y finitos.
- `cv` se estima con el kernel `gaussian` como referencia y el bandwidth
  elegido se aplica luego a los seis kernels.
- El backend KDE se elige automaticamente. Para forzarlo, definir
  `KDE_BACKEND=gpu` o `KDE_BACKEND=cpu` antes de ejecutar el notebook.
- GPU requiere CuPy compatible con CUDA. En esta maquina se valido con
  `cupy-cuda13x` y una RTX 4060.
- La CV del notebook usa una submuestra rapida por defecto. Para una corrida
  mas fina se pueden ajustar `KDE_CV_SUBSAMPLE`, `KDE_CV_FOLDS` y
  `KDE_CV_BW_GRID` antes de ejecutar.
- La carpeta queda limitada a carga de datos, estimacion KDE y figuras finales.
