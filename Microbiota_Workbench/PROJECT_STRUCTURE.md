# Estructura del proyecto

La librería (`Libreria_Python/`) contiene solo las funciones; la GUI de pruebas y los scripts de prueba viven aparte en `Pruebas/`.

```text
Microbiota_Workbench/
├── Libreria_Python/             # librería + backend de la API
│   ├── modules/
│   │   ├── load/
│   │   ├── preprocessing/            # log1p, CLR
│   │   ├── dimensionality_reduction/ # PCA, KPCA, Isomap, MDS, t-SNE, UMAP
│   │   ├── characterization/
│   │   ├── create_df/
│   │   ├── dbscan/
│   │   ├── exploration/
│   │   ├── visualizations/
│   │   ├── kde/
│   │   ├── kruskal_wallis/
│   │   ├── mann_whitney/
│   │   ├── math_agent/
│   │   ├── odds_ratio/
│   │   └── smart_assistant/
│   ├── api_server.py            # API para el frontend Angular
│   ├── Datos/
│   └── .env
└── Pruebas/                     # GUI de pruebas, web, scripts y tests (usan la librería)
    ├── _bootstrap.py            # agrega Libreria_Python al sys.path
    ├── gui/                     # GUI Tkinter
    │   ├── run_gui.py           # lanzador
    │   └── gui_app.py
    ├── web/
    │   └── web_app.py           # web de pruebas
    ├── ollama/                  # utilidades del modelo local
    │   ├── benchmark_ollama.py
    │   └── check_ollama_phi.py
    ├── tests/                   # test_assistant_smoke.py, test_gui_ollama.py, test_math_agent.py
    └── ejemplos/
        └── main.py              # ejemplo de uso de la librería
```

## Ejecutar la GUI de pruebas

```bat
cd Pruebas\gui
python run_gui.py
```

Cada script de `Pruebas/` agrega la carpeta `Pruebas` al `sys.path` e importa `_bootstrap`, de modo que `from modules... import ...` funciona sin instalar la librería.
