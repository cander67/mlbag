"""Bayesian hyperparameter search backed by Optuna. Requires the `bayesian` extra."""

from __future__ import annotations

from typing import Any, Callable, Literal

from mlbag.tuning.base import SearchResult


def bayesian_search(
    objective_fn: Callable[[Any], float] | None = None,
    *,
    estimator: Any = None,
    param_space: dict | None = None,
    X: Any = None,
    y: Any = None,
    n_trials: int = 30,
    direction: Literal["maximize", "minimize"] = "maximize",
    scoring: str | Any = "accuracy",
    cv: int = 5,
    random_state: int = 42,
) -> SearchResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if objective_fn is None:
        if estimator is None or param_space is None or X is None or y is None:
            raise ValueError("bayesian_search requires either objective_fn, or estimator+param_space+X+y")
        objective_fn = _make_sklearn_objective(estimator, param_space, X, y, scoring=scoring, cv=cv)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)

    best_estimator = None
    if estimator is not None and param_space is not None:
        from sklearn.base import clone

        best_estimator = clone(estimator).set_params(**study.best_params)
        best_estimator.fit(X, y)

    return SearchResult(
        best_params=study.best_params,
        best_score=study.best_value,
        best_estimator=best_estimator,
        trials=study.trials_dataframe(),
        search_type="bayesian",
    )


def _make_sklearn_objective(estimator: Any, param_space: dict, X: Any, y: Any, *, scoring: Any, cv: int) -> Callable:
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    def objective(trial: Any) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in param_space.items()}
        candidate = clone(estimator).set_params(**params)
        scores = cross_val_score(candidate, X, y, scoring=scoring, cv=cv)
        return float(scores.mean())

    return objective


def _suggest(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, tuple):
        kind = spec[0]
        if kind == "categorical":
            return trial.suggest_categorical(name, spec[1])
        if kind == "int":
            _, low, high = spec[:3]
            return trial.suggest_int(name, low, high)
        if kind == "float":
            _, low, high, *rest = spec
            return trial.suggest_float(name, low, high, log=bool(rest) and rest[0] == "log")
    raise ValueError(f"Unrecognized param spec: {spec!r}")
