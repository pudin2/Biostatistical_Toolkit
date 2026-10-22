umap = None
HAS_UMAP = None
UMAP_IMPORT_ERROR = None


def _load_umap():
    global umap, HAS_UMAP, UMAP_IMPORT_ERROR

    if HAS_UMAP is not True:
        try:
            import umap as _umap
            umap = _umap
            HAS_UMAP = True
            UMAP_IMPORT_ERROR = None
        except Exception as exc:
            umap = None
            HAS_UMAP = False
            UMAP_IMPORT_ERROR = exc

    if HAS_UMAP is not True:
        raise ImportError(
            "UMAP no esta disponible en este entorno. "
            "Instala/repara umap-learn o usa otro embedding. "
            f"Detalle: {UMAP_IMPORT_ERROR}"
        )


def reduce_umap(X, n_components=10, random_state=42, **kwargs):
    _load_umap()

    n_samples, n_features = X.shape
    max_components = min(n_components, n_features)

    if max_components < 2:
        raise ValueError("UMAP requiere al menos 2 componentes.")

    n_neighbors = min(kwargs.get("n_neighbors", 15), n_samples - 1)

    model = umap.UMAP(
        n_components=max_components,
        n_neighbors=n_neighbors,
        min_dist=kwargs.get("min_dist", 0.1),
        metric=kwargs.get("metric", "euclidean"),
        random_state=random_state
    )
    return model.fit_transform(X), model
