"""Training curve plotting."""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure


def plot_training_curves(
    history: dict[str, list[float]],
    *,
    metrics: list[str] | None = None,
    ax: Any = None,
) -> Figure:
    import matplotlib.pyplot as plt

    series_names = metrics if metrics is not None else list(history.keys())

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for name in series_names:
        values = history.get(name)
        if values is None:
            continue
        ax.plot(range(1, len(values) + 1), values, label=name)

    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    return fig
