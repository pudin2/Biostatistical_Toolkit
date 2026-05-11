# Kernel_Tests

Etapa KDE del proyecto en un solo notebook autocontenido.

## Estructura

```text
Kernel_Tests/
  KDE.ipynb
  Kernels_Formas_Funcionales.pdf
  README.md
  requirements.txt
```

## Arquitectura

`KDE.ipynb` sigue la misma estructura del resto del proyecto:

1. imports;
2. funciones de carga con `load_dataframe_from_path` y `load_multiple_dataframes`;
3. carga de archivos con `dfs = load_multiple_dataframes()`;
4. definicion de funciones auxiliares;
5. llamada final al flujo con `kde_from_loaded(...)`.

## Flujo

El notebook hace todo el flujo KDE en una sola ejecucion:

- carga el DataFrame seleccionado;
- toma valores OTU positivos;
- evalua la grilla logaritmica;
- estima bandwidths por kernel mediante validacion cruzada con expansion
  automatica del rango si el mejor valor cae en un borde;
- calcula KDE con un `h` comun;
- calcula KDE con el mejor `h` particular de cada kernel;
- permite pruebas particulares de bandwidth por kernel cuando se entrega un
  diccionario en `test_kernel_bandwidths`.

El archivo `Kernels_Formas_Funcionales.pdf` documenta las formas funcionales
de los kernels usados en el notebook.

## Figuras

El notebook genera solo las figuras necesarias:

- seleccion de bandwidth por kernel;
- evaluacion de grilla;
- KDE de los 12 kernels con `h` comun;
- KDE por kernel con el mismo `h` comun;
- KDE de los 12 kernels con `h` particular;
- pruebas particulares de bandwidth por kernel, solo cuando se define
  `test_kernel_bandwidths`.

No se incluye tabla de resumen de filas, columnas o valores positivos, porque
esa caracterizacion corresponde a otro modulo del proyecto. La tabla de grilla
solo resume el rango de la grilla y el `grid_size` usado.

## Kernels incluidos

- `gaussian`
- `epanechnikov`
- `tophat`
- `exponential`
- `linear`
- `cosine`
- `quartic`
- `triweight`
- `tricube`
- `logistic`
- `sigmoid`
- `cauchy`

## Parametros principales

La ultima celda controla el flujo:

```python
kde_outputs = kde_from_loaded(
    dfs=dfs,
    data_df_name="otu_data_converted",
    grid_size=1000,
    cv_subsample=1000,
    cv_folds=3,
    cv_bw_grid=8,
    min_bandwidth=1.0,
    cv_max_expansions=4,
    test_kernel_bandwidths=None,
    verbose=True,
)
```

`min_bandwidth` evita soluciones artificiales con `h` demasiado pequeno en
datos de conteos repetidos. Si una matriz continua requiere valores menores,
se puede reducir ese parametro en la llamada final.

Para cambiar pruebas particulares de bandwidth, reemplazar
`test_kernel_bandwidths=None` por un diccionario con los 12 kernels. Si queda
en `None`, esa figura no se calcula.
