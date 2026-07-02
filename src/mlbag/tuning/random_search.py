"""Randomized search wrapper producing the shared trials DataFrame schema."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV

from mlbag.tuning.base import SearchResult, cv_results_to_trials


def random_search(
    estimator: Any,
    param_distributions: dict,
    X: Any,
    y: Any,
    *,
    scoring: str | Any = "accuracy",
    cv: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
) -> SearchResult:
    search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return SearchResult(
        best_params=search.best_params_,
        best_score=search.best_score_,
        best_estimator=search.best_estimator_,
        trials=cv_results_to_trials(search.cv_results_),
        search_type="random",
    )
