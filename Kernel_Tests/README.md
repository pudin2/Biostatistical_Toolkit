# Kernel_Tests

Etapa modular para estimar densidades KDE sobre valores OTU positivos.
La carpeta queda organizada como notebooks independientes y autocontenidos.

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
   - Carga los valores positivos.
   - Resume el volumen de datos.
   - Propone una grilla logaritmica.
   - Muestra el valor `grid_size` que puede copiarse al notebook de kernels.

2. `Estimacion_Bandwidths.ipynb`
   - Calcula bandwidths por `cv`, `scott` y `silverman`.
   - Compara visualmente las tres alternativas.
   - Muestra `selected_method` y `selected_bandwidth` para copiarlos al
     notebook de kernels.

3. `Calculo_Kernels.ipynb`
   - Usa el bandwidth y la grilla elegidos.
   - Evalua los seis kernels KDE.
   - Valida que la metodo rapido sea numericamente cercana a la referencia.
   - Muestra tablas resumen y graficas finales dentro del notebook.

Los notebooks no se comunican automaticamente. Los valores relevantes se
copian manualmente cuando se quiera encadenar una evaluacion con otra.

## Datos de prueba

Cada notebook tiene una variable:

```python
USE_SYNTHETIC_DATA = False
```

Cambiarla a `True` permite probar el flujo con datos sinteticos pequenos.

## Kernels disponibles

- `gaussian`
- `epanechnikov`
- `tophat`
- `exponential`
- `linear`
- `cosine`

## Tiempos de referencia

En la medicion local con los datos reales actuales:

- Estimacion de bandwidth por validacion cruzada: ~18.4 s.
- Evaluacion final de los seis kernels con grilla de 1000 puntos: ~2.0 s.

Estos tiempos pueden variar segun el equipo y la configuracion de cada
notebook.
