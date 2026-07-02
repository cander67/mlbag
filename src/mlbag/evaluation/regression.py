"""Regression evaluation: metrics and residual plot persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

if TYPE_CHECKING:
    from mlbag.tracking import RunManager


def evaluate_regressor(
    y_true: Any,
    y_pred: Any,
    *,
    run: "RunManager | None" = None,
    split_name: str = "test",
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }

    if run is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(f"{split_name} residuals")
        run.save_plot(fig, f"{split_name}_residuals")
        plt.close(fig)

        run.save_metrics(metrics, filename=f"{split_name}_metrics.json")

    return metrics
