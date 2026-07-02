from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from mlbag.config import MLBagConfig


class NetConfig(BaseModel):
    hidden_size: int = 64


class DummyConfig(MLBagConfig):
    learning_rate: float = 1e-3
    batch_size: int = 32
    net: NetConfig = NetConfig()


def test_yaml_round_trip(tmp_path):
    config = DummyConfig(run_name="test", learning_rate=0.01, batch_size=64)
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = DummyConfig.from_yaml(path)
    assert loaded == config


def test_from_cli_overrides_sets_top_level_field():
    base = DummyConfig(run_name="base", learning_rate=0.01)
    overridden = DummyConfig.from_cli_overrides(base, {"learning_rate": "0.5", "batch_size": "128"})
    assert overridden.learning_rate == 0.5
    assert overridden.batch_size == 128
    assert overridden.run_name == "base"


def test_from_cli_overrides_sets_nested_field():
    base = DummyConfig(run_name="base")
    overridden = DummyConfig.from_cli_overrides(base, {"net.hidden_size": "128"})
    assert overridden.net.hidden_size == 128


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        DummyConfig(run_name="test", unknown_field=1)


def test_output_dir_defaults_to_path():
    config = DummyConfig(run_name="test")
    assert isinstance(config.output_dir, Path)
    assert config.output_dir == Path("artifacts")
