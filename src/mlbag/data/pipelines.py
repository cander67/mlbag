"""Model pipeline factory, generalizing part_4.ipynb's make_model_pipeline. Requires `sklearn`."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def make_model_pipeline(
    model: Any,
    *,
    selector: Any = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    scaler: str | None = "standard",
) -> Pipeline:
    steps: list[tuple[str, Any]] = []

    if numeric_features or categorical_features:
        transformers = []
        if numeric_features:
            numeric_step = _SCALERS[scaler]() if scaler else "passthrough"
            transformers.append(("numeric", numeric_step, numeric_features))
        if categorical_features:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features))
        steps.append(("preprocess", ColumnTransformer(transformers)))
    elif scaler:
        steps.append(("scaler", _SCALERS[scaler]()))

    if selector is not None:
        steps.append(("selector", selector))

    steps.append(("model", model))
    return Pipeline(steps)
