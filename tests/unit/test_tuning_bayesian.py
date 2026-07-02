import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlbag.tuning import SearchResult, optimize
from mlbag.tuning.bayesian import bayesian_search

pytestmark = pytest.mark.slow

pytest.importorskip("optuna")


def _toy_dataset():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y


def test_bayesian_search_sklearn_mode_returns_search_result():
    X, y = _toy_dataset()
    result = bayesian_search(
        estimator=LogisticRegression(max_iter=200),
        param_space={"C": ("float", 0.01, 10, "log")},
        X=X,
        y=y,
        n_trials=5,
        cv=3,
        random_state=0,
    )
    assert isinstance(result, SearchResult)
    assert result.search_type == "bayesian"
    assert len(result.trials) == 5
    assert result.best_estimator is not None


def test_bayesian_search_custom_objective_fn():
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return -((x - 2) ** 2)

    result = bayesian_search(objective, n_trials=15, direction="maximize")
    assert result.best_params["x"] == pytest.approx(2, abs=1.0)


def test_bayesian_search_requires_objective_or_sklearn_args():
    with pytest.raises(ValueError):
        bayesian_search()


def test_optimize_dispatches_to_bayesian():
    X, y = _toy_dataset()
    result = optimize(
        LogisticRegression(max_iter=200),
        {"C": ("float", 0.01, 10, "log")},
        X,
        y,
        method="bayesian",
        n_iter=5,
        cv=3,
    )
    assert result.search_type == "bayesian"
