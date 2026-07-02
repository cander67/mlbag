"""Train/val/test splitting helpers. Requires the `sklearn` extra."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import train_test_split


def stratified_split(
    X: Any,
    y: Any,
    *,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_state: int = 42,
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if val_size is None:
        return X_train, X_test, y_train, y_test

    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=relative_val_size, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
