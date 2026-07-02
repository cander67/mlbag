import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlbag.tracking import RunManager
from mlbag.tuning import SearchResult, optimize


def _toy_dataset():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y


def test_optimize_grid_search_finds_param_in_grid():
    X, y = _toy_dataset()
    result = optimize(LogisticRegression(max_iter=200), {"C": [0.1, 1.0, 10.0]}, X, y, method="grid", cv=3)
    assert isinstance(result, SearchResult)
    assert result.best_params["C"] in [0.1, 1.0, 10.0]
    assert len(result.trials) == 3


def test_optimize_random_search_respects_n_iter():
    X, y = _toy_dataset()
    result = optimize(
        LogisticRegression(max_iter=200),
        {"C": ("float", 0.01, 10, "log")},
        X,
        y,
        method="random",
        cv=3,
        n_iter=4,
        random_state=0,
    )
    assert result.search_type == "random"
    assert len(result.trials) == 4


def test_optimize_persists_artifacts_when_run_given(tmp_path):
    X, y = _toy_dataset()
    run = RunManager(base_dir=tmp_path, run_id="tune_run")
    optimize(LogisticRegression(max_iter=200), {"C": [0.1, 1.0]}, X, y, method="grid", cv=3, run=run)

    assert (run.metrics_dir / "trials.csv").exists()
    assert (run.metrics_dir / "best_params.json").exists()
    assert run.latest_checkpoint("best_model") is not None


def test_optimize_unknown_method_raises():
    X, y = _toy_dataset()
    with pytest.raises(ValueError):
        optimize(LogisticRegression(), {"C": [1.0]}, X, y, method="bogus")
