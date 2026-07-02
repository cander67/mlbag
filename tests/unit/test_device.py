import pytest

from mlbag.device import device_summary, get_torch_device


def test_device_summary_returns_dict():
    summary = device_summary()
    assert isinstance(summary, dict)


def test_get_torch_device_cpu_fallback():
    pytest.importorskip("torch")
    device = get_torch_device(prefer="cpu")
    assert str(device) == "cpu"
