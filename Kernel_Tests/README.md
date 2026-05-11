# Kernel_Tests

Etapa modular para estimar densidades KDE sobre valores OTU positivos.

## Estructura

```text
Kernel_Tests/
  Estimacion_Grid.ipynb
  Estimacion_Bandwidths.ipynb
  Calculo_Kernels.ipynb
  README.md
  requirements.txt
```

## Flujo

1. `Estimacion_Grid.ipynb`
   - Carga los valores positivos desde `DATA_PATH`.
   - Resume el volumen de datos.
   - Propone y grafica una grilla logaritmica.
   - Muestra el `grid_size` que puede copiarse al notebook de kernels.

2. `Estimacion_Bandwidths.ipynb`
   - Calcula bandwidths por kernel mediante validacion cruzada.
   - Tambien muestra las referencias globales de Scott y Silverman.
   - Grafica la KDE usando el bandwidth optimo de cada kernel.
   - Muestra el diccionario `kernel_bandwidths` para copiarlo al notebook final.

3. `Calculo_Kernels.ipynb`
   - Usa `DATA_PATH`, `GRID_SIZE` y `kernel_bandwidths`.
   - Evalua los seis kernels KDE.
   - Valida que el metodo rapido sea numericamente cercano a la referencia.
   - Muestra graficas comparativas y graficas individuales.
   - Incluye una seccion de pruebas particulares donde puede cambiarse el
     bandwidth de cada kernel sin alterar la evaluacion principal.

Los notebooks no se comunican automaticamente. Los valores relevantes se
copian manualmente cuando se quiera encadenar una evaluacion con otra.

## Parametros principales

Cada notebook usa una ruta relativa editable:

```python
DATA_PATH = "../Datos/otu_data_converted.csv"
```

Para usar otro archivo, basta con reemplazar ese texto por otra ruta relativa.

En los notebooks de grid y kernels, el tamano de grilla se define siempre con
un valor explicito:

```python
GRID_SIZE = 1000
```

En `Calculo_Kernels.ipynb`, los bandwidths principales se configuran asi:

```python
kernel_bandwidths = {
    "gaussian": 265.702434254,
    "epanechnikov": 265.702434254,
    "tophat": 265.702434254,
    "exponential": 265.702434254,
    "linear": 265.702434254,
    "cosine": 265.702434254,
}
```

Y las pruebas particulares se controlan con:

```python
test_kernel_bandwidths = {
    "gaussian": kernel_bandwidths["gaussian"],
    "epanechnikov": kernel_bandwidths["epanechnikov"],
    "tophat": kernel_bandwidths["tophat"],
    "exponential": kernel_bandwidths["exponential"],
    "linear": kernel_bandwidths["linear"],
    "cosine": kernel_bandwidths["cosine"],
}
```

## Kernels disponibles

- `gaussian`
- `epanechnikov`
- `tophat`
- `exponential`
- `linear`
- `cosine`

## Tiempos de referencia

En la medicion local con los datos reales actuales:

- Estimacion de bandwidth por validacion cruzada para los seis kernels,
  usando `CV_SUBSAMPLE = 1000`, `CV_FOLDS = 3` y `CV_BW_GRID = 8`: ~1.3 s.
- Evaluacion final de los seis kernels con grilla de 1000 puntos: ~2.0 s.

Si se aumenta `CV_SUBSAMPLE`, `CV_FOLDS` o `CV_BW_GRID`, la estimacion de
bandwidths sera mas fina, pero tambien mas lenta.
