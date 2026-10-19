from sklearn.decomposition import KernelPCA


def reduce_kpca(X, n_components=10, random_state=42, **kwargs):
    n_samples, n_features = X.shape
    max_components = min(n_components, n_samples - 1, n_features)

    if max_components < 2:
        raise ValueError("No hay suficientes dimensiones para KernelPCA.")

    model = KernelPCA(
        n_components=max_components,
        kernel=kwargs.get("kernel", "rbf"),
        gamma=kwargs.get("gamma", None)
    )
    return model.fit_transform(X), model
