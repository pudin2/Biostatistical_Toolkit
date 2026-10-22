from .isomap import reduce_isomap
from .kpca import reduce_kpca
from .mds import reduce_mds
from .pca import reduce_pca
from .tsne import reduce_tsne
from .umap_reducer import reduce_umap

REDUCERS = {
    "pca": reduce_pca,
    "kpca": reduce_kpca,
    "isomap": reduce_isomap,
    "mds": reduce_mds,
    "tsne": reduce_tsne,
    "umap": reduce_umap,
}


def apply_embedding(
    X_scaled,
    method="pca",
    n_components=10,
    random_state=42,
    **kwargs
):
    """Aplica la reduccion de dimensionalidad `method` y devuelve (X_embed, modelo)."""
    method = str(method).lower()

    if method == "none":
        return X_scaled, None

    if X_scaled.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 filas para aplicar embedding.")

    if method not in REDUCERS:
        raise ValueError(f"Embedding no soportado: {method}")

    return REDUCERS[method](
        X_scaled,
        n_components=n_components,
        random_state=random_state,
        **kwargs
    )
