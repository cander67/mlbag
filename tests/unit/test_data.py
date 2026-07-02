import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlbag.data import get_top_ngram, make_model_pipeline, stratified_split


def test_stratified_split_preserves_class_proportions():
    y = np.array([0] * 80 + [1] * 20)
    X = np.arange(len(y)).reshape(-1, 1)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=42)
    assert abs(y_train.mean() - y.mean()) < 0.05
    assert abs(y_test.mean() - y.mean()) < 0.05


def test_stratified_split_with_val_size_returns_three_way_split():
    y = np.array([0] * 50 + [1] * 50)
    X = np.arange(len(y)).reshape(-1, 1)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(y_train) + len(y_val) + len(y_test) == len(y)
    assert len(y_test) == 20


def test_stratified_split_reproducible_with_random_state():
    y = np.array([0] * 50 + [1] * 50)
    X = np.arange(len(y)).reshape(-1, 1)
    split_a = stratified_split(X, y, random_state=1)
    split_b = stratified_split(X, y, random_state=1)
    assert (split_a[0] == split_b[0]).all()


def test_make_model_pipeline_fits_mixed_dataframe():
    df = pd.DataFrame({"age": [20, 30, 40, 50], "city": ["nyc", "sf", "nyc", "la"]})
    y = [0, 1, 0, 1]
    pipeline = make_model_pipeline(
        LogisticRegression(), numeric_features=["age"], categorical_features=["city"]
    )
    pipeline.fit(df, y)
    assert len(pipeline.predict(df)) == len(y)


def test_make_model_pipeline_without_feature_lists_uses_plain_scaler():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = [0, 1, 0, 1]
    pipeline = make_model_pipeline(LogisticRegression())
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(y)


def test_get_top_ngram_known_answer():
    corpus = ["cat dog", "cat cat", "dog dog dog"]
    top = get_top_ngram(corpus, n=1, top_k=2)
    words = [word for word, _ in top]
    assert "dog" in words
    assert "cat" in words


def test_get_top_ngram_bigrams():
    corpus = ["new york city", "new york new york"]
    top = get_top_ngram(corpus, n=2, top_k=1)
    assert top[0][0] == "new york"
