"""Generic single-model supervised training loop for Keras, mirroring torch_loop's API
shape. Requires the `tensorflow` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import keras
import numpy as np

if TYPE_CHECKING:
    from mlbag.tracking import RunManager


class RunManagerCallback(keras.callbacks.Callback):
    def __init__(
        self,
        run: "RunManager",
        *,
        val_data: Any = None,
        class_names: list[str] | None = None,
        checkpoint_every: int | None = None,
    ) -> None:
        super().__init__()
        self.run = run
        self.val_data = val_data
        self.class_names = class_names
        self.checkpoint_every = checkpoint_every

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        metrics = {"epoch": epoch + 1, **{k: float(v) for k, v in logs.items()}}

        if self.val_data is not None and self.class_names is not None:
            from mlbag.evaluation import evaluate_classifier

            X_val, y_val = self.val_data
            y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
            report = evaluate_classifier(y_val, y_pred, labels=self.class_names)
            metrics["val_macro_f1"] = report["macro avg"]["f1-score"]

        self.run.save_metrics(metrics, filename="history.jsonl", append=True)

        if self.checkpoint_every and (epoch + 1) % self.checkpoint_every == 0:
            self.run.save_checkpoint(self.model, name="model", epoch=epoch + 1)


def make_callbacks(
    run: "RunManager",
    *,
    val_data: Any = None,
    class_names: list[str] | None = None,
    early_stopping: bool = True,
    patience: int = 3,
    checkpoint_every: int | None = None,
) -> list:
    callbacks: list = [
        RunManagerCallback(run, val_data=val_data, class_names=class_names, checkpoint_every=checkpoint_every)
    ]
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )
    return callbacks


def run_training(
    model: Any,
    train_data: Any,
    val_data: Any,
    *,
    epochs: int,
    run: "RunManager | None" = None,
    class_names: list[str] | None = None,
    checkpoint_every: int | None = None,
    early_stopping: bool = True,
    patience: int = 3,
    **fit_kwargs: Any,
) -> dict:
    callbacks = list(fit_kwargs.pop("callbacks", []))
    if run is not None:
        callbacks.extend(
            make_callbacks(
                run,
                val_data=val_data,
                class_names=class_names,
                early_stopping=early_stopping,
                patience=patience,
                checkpoint_every=checkpoint_every,
            )
        )
    elif early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )

    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        history = model.fit(
            X_train, y_train, validation_data=val_data, epochs=epochs, callbacks=callbacks, **fit_kwargs
        )
    else:
        history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks, **fit_kwargs)

    if run is not None:
        run.save_metrics(
            {k: [float(v) for v in vals] for k, vals in history.history.items()}, filename="final_history.json"
        )
        run.save_checkpoint(model, name="best_model")

    return history.history
