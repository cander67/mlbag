"""Local file-based run tracking: run directories, config/metrics, checkpoints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlbag.config import MLBagConfig

_SUBDIRS = ("checkpoints", "plots", "metrics", "logs")


@dataclass
class RunSummary:
    run_id: str
    run_dir: Path
    status: str | None
    config: dict[str, Any] | None


class RunManager:
    def __init__(
        self,
        base_dir: str | Path,
        run_id: str | None = None,
        *,
        project: str | None = None,
        resume: bool = False,
    ) -> None:
        base_dir = Path(base_dir)
        self.run_id = self.sanitize_run_id(run_id) if run_id else self._generate_run_id(project)
        self.base_dir = base_dir
        self._run_dir = base_dir / self.run_id

        if resume and not self._run_dir.exists():
            raise FileNotFoundError(f"Cannot resume: run directory {self._run_dir} does not exist")

        for sub in _SUBDIRS:
            (self._run_dir / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _generate_run_id(project: str | None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return RunManager.sanitize_run_id(f"{timestamp}_{project or 'run'}")

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def checkpoints_dir(self) -> Path:
        return self._run_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self._run_dir / "plots"

    @property
    def metrics_dir(self) -> Path:
        return self._run_dir / "metrics"

    # -- config -----------------------------------------------------------

    def save_config(self, config: "MLBagConfig") -> Path:
        path = self._run_dir / "config.json"
        path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))
        return path

    def load_config(self, config_cls: type["MLBagConfig"]) -> "MLBagConfig":
        path = self._run_dir / "config.json"
        return config_cls.model_validate_json(path.read_text())

    # -- metrics ------------------------------------------------------------

    def save_metrics(self, metrics: dict, filename: str = "metrics.json", *, append: bool = False) -> Path:
        path = self.metrics_dir / filename
        if append:
            with open(path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        else:
            path.write_text(json.dumps(metrics, indent=2))
        return path

    def load_metrics(self, filename: str = "metrics.json") -> dict | list[dict]:
        path = self.metrics_dir / filename
        text = path.read_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]

    # -- checkpoints ----------------------------------------------------------

    def save_checkpoint(
        self,
        obj: Any,
        name: str = "model",
        *,
        epoch: int | None = None,
        optimizer: Any = None,
        scheduler: Any = None,
        extra: dict | None = None,
    ) -> Path:
        suffix = f"_epoch{epoch}" if epoch is not None else ""

        try:
            from sklearn.base import BaseEstimator
        except ImportError:
            BaseEstimator = None  # type: ignore[assignment, misc]

        if BaseEstimator is not None and isinstance(obj, BaseEstimator):
            import joblib

            path = self.checkpoints_dir / f"{name}{suffix}.joblib"
            joblib.dump(obj, path)
            return path

        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        if torch is not None and isinstance(obj, torch.nn.Module):
            path = self.checkpoints_dir / f"{name}{suffix}.pt"
            payload: dict[str, Any] = {"model_state_dict": obj.state_dict(), "epoch": epoch, "extra": extra}
            if optimizer is not None:
                payload["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                payload["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(payload, path)
            return path

        try:
            import keras
        except ImportError:
            keras = None  # type: ignore[assignment]

        if keras is not None and isinstance(obj, keras.Model):
            path = self.checkpoints_dir / f"{name}{suffix}.keras"
            obj.save(path)
            return path

        raise TypeError(f"save_checkpoint: unsupported object type {type(obj)!r}")

    def load_checkpoint(
        self,
        obj_or_cls: Any,
        path: str | Path | None = None,
        *,
        device: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> Any:
        if path is None:
            found = self.latest_checkpoint()
            if found is None:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoints_dir}")
            path = found
        path = Path(path)

        if path.suffix == ".joblib":
            import joblib

            return joblib.load(path)

        if path.suffix == ".pt":
            import torch

            payload = torch.load(path, map_location=device)
            obj_or_cls.load_state_dict(payload["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in payload:
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            if device is not None:
                obj_or_cls.to(device)
            return obj_or_cls

        if path.suffix == ".keras":
            import keras

            return keras.models.load_model(path)

        raise ValueError(f"load_checkpoint: unrecognized checkpoint suffix {path.suffix!r}")

    def latest_checkpoint(self, name: str = "model") -> Path | None:
        candidates = sorted(self.checkpoints_dir.glob(f"{name}*"), key=lambda p: p.stat().st_mtime)
        return candidates[-1] if candidates else None

    # -- plots / tables -----------------------------------------------------

    def save_plot(self, fig: Any, name: str) -> Path:
        if not name.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
            name = f"{name}.png"
        path = self.plots_dir / name
        fig.savefig(path, bbox_inches="tight")
        return path

    def save_dataframe(self, df: Any, name: str) -> Path:
        if not name.endswith(".csv"):
            name = f"{name}.csv"
        path = self.metrics_dir / name
        df.to_csv(path, index=False)
        return path

    # -- bookkeeping ----------------------------------------------------------

    def mark_complete(self, status: str = "success") -> None:
        (self._run_dir / "status.json").write_text(json.dumps({"status": status}))

    @staticmethod
    def sanitize_run_id(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
        return sanitized.strip("_") or "run"

    @staticmethod
    def list_runs(base_dir: str | Path) -> list[RunSummary]:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return []

        summaries = []
        for run_dir in sorted(base_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text())["status"] if status_path.exists() else None
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else None
            summaries.append(RunSummary(run_id=run_dir.name, run_dir=run_dir, status=status, config=config))
        return summaries
