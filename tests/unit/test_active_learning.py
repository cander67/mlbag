import numpy as np
import pytest

from mlbag.active_learning import (
    query_by_committee,
    run_active_learning_loop,
    uncertainty_sampling,
)


def test_uncertainty_sampling_picks_highest_uncertainty_indices():
    uncertainty = np.array([0.1, 0.9, 0.3, 0.7])
    idx = uncertainty_sampling(uncertainty, k=2)
    assert set(idx.tolist()) == {1, 3}


def test_uncertainty_sampling_raises_if_k_exceeds_pool_size():
    with pytest.raises(ValueError):
        uncertainty_sampling(np.array([0.1, 0.2]), k=5)


def test_query_by_committee_picks_highest_disagreement():
    predictions = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 5.0, 1.1],
            [1.0, -3.0, 0.9],
        ]
    )
    idx = query_by_committee(predictions, k=1)
    assert idx[0] == 1


class _MeanModel:
    """Trivial fake estimator: predicts the mean of y seen during fit."""

    def fit(self, X, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def _score_by_first_feature(model, pool_X):
    # Deterministic "uncertainty": the first feature value, so tests know exactly how
    # many/which points get queried without depending on model internals.
    return pool_X[:, 0]


def test_run_active_learning_loop_grows_labeled_set_and_shrinks_pool():
    rng = np.random.RandomState(0)
    pool_X = rng.rand(10, 2)
    initial_X = rng.rand(3, 2)
    initial_y = rng.rand(3)

    history = run_active_learning_loop(
        model_fn=_MeanModel,
        acquisition_fn=_score_by_first_feature,
        pool_X=pool_X,
        oracle_fn=lambda queried_X: queried_X[:, 0] * 2,
        initial_X=initial_X,
        initial_y=initial_y,
        n_iterations=3,
        query_size=2,
    )

    assert history.n_labeled == [5, 7, 9]
    assert history.iteration == [0, 1, 2]
    assert all(len(q) == 2 for q in history.queried_indices)


def test_run_active_learning_loop_stops_when_pool_exhausted():
    rng = np.random.RandomState(1)
    pool_X = rng.rand(4, 2)
    initial_X = rng.rand(2, 2)
    initial_y = rng.rand(2)

    history = run_active_learning_loop(
        model_fn=_MeanModel,
        acquisition_fn=_score_by_first_feature,
        pool_X=pool_X,
        oracle_fn=lambda q: q[:, 0],
        initial_X=initial_X,
        initial_y=initial_y,
        n_iterations=10,
        query_size=2,
    )

    # pool has 4 points, queried 2 at a time -> the loop can only run twice
    assert len(history.iteration) == 2
    assert history.n_labeled[-1] == 2 + 4


def test_run_active_learning_loop_calls_eval_fn_and_records_score():
    rng = np.random.RandomState(2)
    pool_X = rng.rand(6, 2)
    initial_X = rng.rand(2, 2)
    initial_y = rng.rand(2)

    history = run_active_learning_loop(
        model_fn=_MeanModel,
        acquisition_fn=_score_by_first_feature,
        pool_X=pool_X,
        oracle_fn=lambda q: q[:, 0],
        initial_X=initial_X,
        initial_y=initial_y,
        n_iterations=2,
        query_size=1,
        eval_fn=lambda model: model.value,
    )

    assert all(isinstance(s, float) for s in history.score)


def test_run_active_learning_loop_logs_metrics_when_run_provided():
    rng = np.random.RandomState(3)
    pool_X = rng.rand(4, 2)
    initial_X = rng.rand(2, 2)
    initial_y = rng.rand(2)

    class _FakeRun:
        def __init__(self):
            self.calls = []

        def save_metrics(self, metrics, filename, append):
            self.calls.append((metrics, filename, append))

    run = _FakeRun()
    run_active_learning_loop(
        model_fn=_MeanModel,
        acquisition_fn=_score_by_first_feature,
        pool_X=pool_X,
        oracle_fn=lambda q: q[:, 0],
        initial_X=initial_X,
        initial_y=initial_y,
        n_iterations=2,
        query_size=1,
        run=run,
    )

    assert len(run.calls) == 2
    assert run.calls[0][1] == "active_learning_history.jsonl"
    assert run.calls[0][2] is True
