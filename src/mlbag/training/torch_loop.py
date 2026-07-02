"""Generic single-model supervised training loop for PyTorch."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from mlbag.tracking import RunManager


@dataclass
class EarlyStopping:
    patience: int = 5
    mode: Literal["min", "max"] = "min"
    min_delta: float = 0.0
    _best: float | None = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def step(self, value: float) -> bool:
        """Update with a new metric value; return True if training should stop."""
        improved = (
            self._best is None
            or (self.mode == "min" and value < self._best - self.min_delta)
            or (self.mode == "max" and value > self._best + self.min_delta)
        )
        if improved:
            self._best = value
            self._count = 0
        else:
            self._count += 1
        return self._count >= self.patience


def run_training(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    criterion: Any,
    *,
    epochs: int,
    device: Any = None,
    scheduler: Any = None,
    run: "RunManager | None" = None,
    checkpoint_every: int | None = None,
    early_stopping: EarlyStopping | None = None,
    metric_fn: Callable[[list, list], dict] | None = None,
) -> dict:
    if device is None:
        from mlbag.device import get_torch_device

        device = get_torch_device()
    model.to(device)

    history: dict[str, list] = defaultdict(list)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred = _evaluate(model, val_loader, criterion, device)

        epoch_metrics = {"train_loss": train_loss, "val_loss": val_loss}
        if metric_fn is not None:
            epoch_metrics.update(metric_fn(y_true, y_pred))
        for key, value in epoch_metrics.items():
            history[key].append(value)

        if scheduler is not None:
            scheduler.step()

        if run is not None:
            run.save_metrics({"epoch": epoch, **epoch_metrics}, filename="history.jsonl", append=True)
            if checkpoint_every and epoch % checkpoint_every == 0:
                run.save_checkpoint(model, name="model", epoch=epoch, optimizer=optimizer)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                run.save_checkpoint(model, name="best_model", optimizer=optimizer)

        if early_stopping is not None and early_stopping.step(val_loss):
            break

    return dict(history)


def _train_one_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        n += inputs.size(0)
    return total_loss / n


def _evaluate(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, list, list]:
    import torch

    model.eval()
    total_loss = 0.0
    n = 0
    y_true: list = []
    y_pred: list = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            n += inputs.size(0)
            preds = outputs.argmax(dim=1) if outputs.ndim > 1 and outputs.shape[1] > 1 else outputs
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return total_loss / n, y_true, y_pred
