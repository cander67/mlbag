"""Embedding scatter plots via PCA/t-SNE/UMAP."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_embedding_scatter(
    X: Any,
    labels: Any = None,
    *,
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    ax: Any = None,
    **method_kwargs: Any,
) -> Figure:
    import matplotlib.pyplot as plt

    X = np.asarray(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, **method_kwargs)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, **method_kwargs)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "plot_embedding_scatter(method='umap') requires `pip install mlbag[viz]`"
            ) from exc

        reducer = UMAP(n_components=n_components, **method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    embedded = reducer.fit_transform(X)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    color_values = None
    uniques = None
    if labels is not None:
        codes, uniques = pd.factorize(np.asarray(labels))
        color_values = codes

    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=color_values, cmap="viridis", s=20, edgecolor="k")
    if uniques is not None:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, list(uniques))

    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    fig.tight_layout()
    return fig
