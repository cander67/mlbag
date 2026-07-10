"""Generic pool-based active learning loop, framework-agnostic.

`pool_X`, `initial_X`, `initial_y`, and oracle-returned labels must be (or coerce to)
numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ActiveLearningHistory:
    iteration: list[int] = field(default_factory=list)
    n_labeled: list[int] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    queried_indices: list[list[int]] = field(default_factory=list)


def run_active_learning_loop(
    model_fn: Callable[[], Any],
    acquisition_fn: Callable[[Any, np.ndarray], np.ndarray],
    pool_X: np.ndarray,
    oracle_fn: Callable[[np.ndarray], np.ndarray],
    *,
    initial_X: np.ndarray,
    initial_y: np.ndarray,
    n_iterations: int,
    query_size: int = 1,
    eval_fn: Callable[[Any], float] | None = None,
    run: Any = None,
) -> ActiveLearningHistory:
    """Run a pool-based active learning loop and return its per-iteration history.

    `model_fn()` returns a fresh, unfitted estimator implementing `.fit(X, y)`.
    `acquisition_fn(model, pool_X)` returns a per-point score (higher = query first) —
    it decides internally what "informative" means (predictive std, committee
    disagreement, etc.), so it may call `model.predict` or any model-specific API.
    `oracle_fn(queried_X)` supplies ground truth for newly queried points; it can be a
    lookup table, a real simulation, or a hybrid of both (see
    `mlbag`-consuming projects' `LookupOrComputeOracle`-style wrappers).
    """
    X_labeled = np.asarray(initial_X)
    y_labeled = np.asarray(initial_y)
    pool_X = np.asarray(pool_X)
    remaining_idx = np.arange(len(pool_X))
    history = ActiveLearningHistory()

    model = model_fn()
    model.fit(X_labeled, y_labeled)

    for iteration in range(n_iterations):
        if len(remaining_idx) == 0:
            break

        scores = acquisition_fn(model, pool_X[remaining_idx])
        k = min(query_size, len(remaining_idx))
        local_idx = np.argsort(scores)[::-1][:k]
        queried_idx = remaining_idx[local_idx]

        queried_y = np.asarray(oracle_fn(pool_X[queried_idx]))
        X_labeled = np.concatenate([X_labeled, pool_X[queried_idx]])
        y_labeled = np.concatenate([y_labeled, queried_y])
        remaining_idx = np.setdiff1d(remaining_idx, queried_idx)

        model = model_fn()
        model.fit(X_labeled, y_labeled)

        score = eval_fn(model) if eval_fn is not None else float("nan")
        history.iteration.append(iteration)
        history.n_labeled.append(len(y_labeled))
        history.score.append(score)
        history.queried_indices.append(queried_idx.tolist())

        if run is not None:
            run.save_metrics(
                {"iteration": iteration, "n_labeled": len(y_labeled), "score": score},
                filename="active_learning_history.jsonl",
                append=True,
            )

    return history
