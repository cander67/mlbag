"""Confusion matrix plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure


def plot_confusion_matrix(
    cm: Any,
    labels: list[str],
    *,
    normalize: bool = False,
    title: str | None = None,
    ax: Any = None,
) -> Figure:
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = format(value, fmt) if normalize else str(int(value))
            ax.text(j, i, text, ha="center", va="center", color="white" if value > thresh else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
