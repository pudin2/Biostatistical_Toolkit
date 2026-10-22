from sklearn.manifold import Isomap


def reduce_isomap(X, n_components=10, random_state=42, **kwargs):
    n_samples, n_features = X.shape
    max_components = min(n_components, n_samples - 1, n_features)

    if max_components < 2:
        raise ValueError("No hay suficientes dimensiones para Isomap.")

    n_neighbors = min(kwargs.get("n_neighbors", 10), n_samples - 1)

    model = Isomap(n_components=max_components, n_neighbors=n_neighbors)
    return model.fit_transform(X), model
