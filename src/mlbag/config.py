"""Typed, YAML-backed configuration base class for project configs."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict


class MLBagConfig(BaseModel):
    """Base class for project configs. Subclass and add project-specific fields."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    seed: int = 42
    output_dir: Path = Path("artifacts")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    @classmethod
    def from_cli_overrides(cls, base: Self, overrides: dict[str, str]) -> Self:
        """Apply `--set key.nested=value`-style dot-path overrides on top of `base`."""
        data = base.model_dump(mode="json")
        for dotted_key, value in overrides.items():
            _set_nested(data, dotted_key.split("."), value)
        return cls.model_validate(data)


def _set_nested(data: dict, keys: list[str], value: str) -> None:
    key, *rest = keys
    if not rest:
        data[key] = _coerce(value)
        return
    data.setdefault(key, {})
    _set_nested(data[key], rest, value)


def _coerce(value: str) -> bool | int | float | str:
    """Best-effort coercion of a CLI override string to bool/int/float/str."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value
