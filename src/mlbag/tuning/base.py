"""Shared types and the unified `optimize()` entry point for hyperparameter search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mlbag.tracking import RunManager


@dataclass
class SearchResult:
    best_params: dict
    best_score: float
    best_estimator: Any
    trials: pd.DataFrame
    search_type: str


def cv_results_to_trials(cv_results: dict) -> pd.DataFrame:
    df = pd.DataFrame(cv_results)
    param_cols = [c for c in df.columns if c.startswith("param_")]
    return df[[*param_cols, "mean_test_score", "std_test_score", "rank_test_score"]]


def resolve_param_values(spec: Any, n_points: int = 5) -> list:
    """Expand a `("float"|"int"|"categorical", ...)` spec into a candidate list,
    or pass a plain list through unchanged."""
    if isinstance(spec, list):
        return spec
    if isinstance(spec, tuple):
        kind = spec[0]
        if kind == "categorical":
            return list(spec[1])
        if kind == "int":
            _, low, high = spec[:3]
            count = min(n_points, high - low + 1)
            values = np.linspace(low, high, num=count)
            return sorted({int(round(v)) for v in values})
        if kind == "float":
            _, low, high, *rest = spec
            scale = rest[0] if rest else "linear"
            values = (
                np.logspace(np.log10(low), np.log10(high), num=n_points)
                if scale == "log"
                else np.linspace(low, high, num=n_points)
            )
            return sorted({float(v) for v in values})
    raise ValueError(f"Unrecognized param spec: {spec!r}")


def optimize(
    estimator: Any,
    param_space: dict,
    X: Any,
    y: Any,
    *,
    method: Literal["grid", "random", "bayesian"] = "grid",
    scoring: str | Any = "accuracy",
    cv: int = 5,
    n_iter: int = 20,
    direction: Literal["maximize", "minimize"] = "maximize",
    run: "RunManager | None" = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> SearchResult:
    if method == "grid":
        from mlbag.tuning.grid import grid_search

        param_grid = {name: resolve_param_values(spec) for name, spec in param_space.items()}
        result = grid_search(estimator, param_grid, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)
    elif method == "random":
        from mlbag.tuning.random_search import random_search

        param_distributions = {
            name: resolve_param_values(spec, n_points=20) for name, spec in param_space.items()
        }
        result = random_search(
            estimator,
            param_distributions,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif method == "bayesian":
        from mlbag.tuning.bayesian import bayesian_search

        result = bayesian_search(
            estimator=estimator,
            param_space=param_space,
            X=X,
            y=y,
            n_trials=n_iter,
            direction=direction,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if run is not None:
        run.save_dataframe(result.trials, "trials")
        run.save_metrics(
            {"best_params": result.best_params, "best_score": result.best_score},
            filename="best_params.json",
        )
        if result.best_estimator is not None:
            run.save_checkpoint(result.best_estimator, name="best_model")

    return result
