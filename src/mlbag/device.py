"""Device selection helpers for PyTorch and TensorFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


def get_torch_device(prefer: Literal["cuda", "mps", "cpu"] | None = None) -> "torch.device":
    """Return the best available torch device, or the one requested via `prefer`."""
    import torch

    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_tf_strategy() -> "tf.distribute.Strategy":
    """Return a MirroredStrategy across GPUs if more than one is available, else the default."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


def device_summary() -> dict[str, str]:
    """Human-readable summary of detected devices, for logging into run metadata."""
    summary: dict[str, str] = {}

    try:
        import torch
    except ImportError:
        summary["torch"] = "not installed"
    else:
        summary["torch_cuda_available"] = str(torch.cuda.is_available())
        summary["torch_mps_available"] = str(torch.backends.mps.is_available())

    try:
        import tensorflow as tf
    except ImportError:
        summary["tensorflow"] = "not installed"
    else:
        summary["tensorflow_gpus"] = str(len(tf.config.list_physical_devices("GPU")))

    return summary
