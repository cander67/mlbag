import numpy as np
from matplotlib.figure import Figure

from mlbag.plotting.confusion import plot_confusion_matrix
from mlbag.plotting.curves import plot_training_curves


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
