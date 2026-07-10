"""Acquisition functions for pool-based active learning: rank pool points by how
informative labeling them would be.
"""

from __future__ import annotations

import numpy as np


def uncertainty_sampling(uncertainty: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the `k` pool points with the highest uncertainty (e.g. predictive std)."""
    uncertainty = np.asarray(uncertainty)
    if k > len(uncertainty):
        raise ValueError(f"k={k} exceeds pool size {len(uncertainty)}")
    return np.argsort(uncertainty)[::-1][:k]


def query_by_committee(predictions: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the `k` pool points with the highest disagreement across an ensemble.

    `predictions` has shape `(n_models, n_pool)` of regression predictions; disagreement is
    each point's standard deviation across committee members.
    """
    predictions = np.asarray(predictions)
    disagreement = predictions.std(axis=0)
    return uncertainty_sampling(disagreement, k)
