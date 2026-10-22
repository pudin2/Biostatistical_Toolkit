from sklearn.manifold import TSNE


def reduce_tsne(X, n_components=10, random_state=42, **kwargs):
    n_samples = X.shape[0]

    if n_samples < 3:
        raise ValueError("t-SNE requiere al menos 3 muestras.")

    max_components = min(n_components, 3)

    if max_components < 2:
        raise ValueError("t-SNE requiere al menos 2 componentes.")

    perplexity = kwargs.get("perplexity", 30)

    if perplexity >= n_samples:
        perplexity = max(2, n_samples - 1)

    model = TSNE(
        n_components=max_components,
        random_state=random_state,
        perplexity=perplexity,
        init=kwargs.get("init", "pca"),
        learning_rate=kwargs.get("learning_rate", "auto")
    )
    return model.fit_transform(X), model
