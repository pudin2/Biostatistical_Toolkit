from sklearn.decomposition import PCA


def reduce_pca(X, n_components=10, random_state=42, **kwargs):
    n_samples, n_features = X.shape
    max_components = min(n_components, n_samples - 1, n_features)

    if max_components < 2:
        raise ValueError("No hay suficientes dimensiones para PCA.")

    model = PCA(n_components=max_components, random_state=random_state)
    return model.fit_transform(X), model
