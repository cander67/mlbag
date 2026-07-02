"""Grid search wrapper producing the shared trials DataFrame schema."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV

from mlbag.tuning.base import SearchResult, cv_results_to_trials


def grid_search(
    estimator: Any,
    param_grid: dict,
    X: Any,
    y: Any,
    *,
    scoring: str | Any = "accuracy",
    cv: int = 5,
    n_jobs: int = -1,
) -> SearchResult:
    search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    search.fit(X, y)
    return SearchResult(
        best_params=search.best_params_,
        best_score=search.best_score_,
        best_estimator=search.best_estimator_,
        trials=cv_results_to_trials(search.cv_results_),
        search_type="grid",
    )
