"""Decision region plotting for 2D classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure


def plot_decision_regions(
    clf: Any,
    X: Any,
    y: Any,
    *,
    X_test: Any = None,
    y_test: Any = None,
    title: str | None = None,
    target_names: list[str] | None = None,
    ax: Any = None,
    h: float = 0.03,
) -> Figure:
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[1] != 2:
        raise ValueError("plot_decision_regions only supports 2D feature input")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cmap = plt.get_cmap("viridis", len(np.unique(y)))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=20)
    if X_test is not None:
        X_test = np.asarray(X_test)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker="^", edgecolor="k", s=40)

    if title:
        ax.set_title(title)
    if target_names:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, target_names, title="class")

    fig.tight_layout()
    return fig
