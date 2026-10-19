from sklearn.manifold import MDS


def reduce_mds(X, n_components=10, random_state=42, **kwargs):
    n_samples = X.shape[0]
    max_components = min(n_components, n_samples - 1)

    if max_components < 2:
        raise ValueError("No hay suficientes dimensiones para MDS.")

    model = MDS(
        n_components=max_components,
        random_state=random_state,
        n_init=kwargs.get("n_init", 4),
        max_iter=kwargs.get("max_iter", 300)
    )
    return model.fit_transform(X), model
