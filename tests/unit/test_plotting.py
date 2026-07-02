import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from mlbag.plotting.confusion import plot_confusion_matrix
from mlbag.plotting.curves import plot_training_curves
from mlbag.plotting.decision_boundary import plot_decision_regions
from mlbag.plotting.embeddings import plot_embedding_scatter


def test_plot_confusion_matrix_returns_figure():
    cm = np.array([[5, 1], [2, 8]])
    fig = plot_confusion_matrix(cm, labels=["neg", "pos"])
    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_normalized_does_not_raise():
    cm = np.array([[0, 0], [2, 8]])
    fig = plot_confusion_matrix(cm, labels=["neg", "pos"], normalize=True)
    assert isinstance(fig, Figure)


def test_plot_training_curves_returns_figure():
    history = {"loss": [1.0, 0.5, 0.3], "val_loss": [1.1, 0.6, 0.4]}
    fig = plot_training_curves(history)
    assert isinstance(fig, Figure)


def test_plot_training_curves_filters_by_metric():
    history = {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6], "lr": [0.01, 0.01]}
    fig = plot_training_curves(history, metrics=["loss"])
    assert isinstance(fig, Figure)


def test_plot_decision_regions_returns_figure():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 1, 0, 1])
    clf = LogisticRegression().fit(X, y)
    fig = plot_decision_regions(clf, X, y, target_names=["neg", "pos"])
    assert isinstance(fig, Figure)


def test_plot_decision_regions_rejects_non_2d_input():
    X = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    y = np.array([0, 1])
    clf = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError):
        plot_decision_regions(clf, X, y)


def test_plot_embedding_scatter_pca_returns_figure():
    X = np.random.RandomState(0).rand(20, 5)
    labels = ["a"] * 10 + ["b"] * 10
    fig = plot_embedding_scatter(X, labels, method="pca")
    assert isinstance(fig, Figure)


def test_plot_embedding_scatter_unknown_method_raises():
    X = np.random.RandomState(0).rand(5, 3)
    with pytest.raises(ValueError):
        plot_embedding_scatter(X, method="bogus")


def test_plot_embedding_scatter_umap_missing_dependency_gives_clear_error():
    try:
        import umap  # noqa: F401

        pytest.skip("umap is installed; clear-error path not exercised")
    except ImportError:
        pass

    X = np.random.RandomState(0).rand(5, 3)
    with pytest.raises(ImportError, match=r"mlbag\[viz\]"):
        plot_embedding_scatter(X, method="umap")
