import pytest

pytestmark = pytest.mark.slow

keras = pytest.importorskip("keras")

from mlbag.tracking import RunManager  # noqa: E402
from mlbag.training.keras_loop import make_callbacks, run_training  # noqa: E402


def _tiny_data():
    import numpy as np

    rng = np.random.RandomState(0)
    X_train = rng.rand(20, 4).astype("float32")
    y_train = rng.randint(0, 2, size=20)
    X_val = rng.rand(10, 4).astype("float32")
    y_val = rng.randint(0, 2, size=10)
    return (X_train, y_train), (X_val, y_val)


def _tiny_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def test_run_training_completes_and_writes_checkpoint(tmp_path):
    train_data, val_data = _tiny_data()
    model = _tiny_model()
    run = RunManager(base_dir=tmp_path, run_id="keras_smoke")

    history = run_training(
        model,
        train_data,
        val_data,
        epochs=2,
        run=run,
        checkpoint_every=1,
        class_names=["neg", "pos"],
        early_stopping=False,
        verbose=0,
    )

    assert "loss" in history
    assert len(history["loss"]) == 2
    assert run.latest_checkpoint("model") is not None
    assert run.latest_checkpoint("best_model") is not None

    history_lines = run.load_metrics("history.jsonl")
    assert len(history_lines) == 2
    assert "val_macro_f1" in history_lines[0]


def test_make_callbacks_returns_early_stopping_when_enabled(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="cb_run")
    callbacks = make_callbacks(run, early_stopping=True, patience=2)
    assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks)
