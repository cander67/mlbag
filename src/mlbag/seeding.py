"""Reproducibility helpers: seed every RNG a project might touch."""

from __future__ import annotations

import os
import random

import numpy as np

_last_seed: int | None = None


def seed_everything(seed: int = 42, *, deterministic_torch: bool = False) -> None:
    """Seed random, numpy, and (if installed) torch/tensorflow with the same seed."""
    global _last_seed

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)

    _last_seed = seed


def get_seed_state() -> dict[str, object]:
    """Return the last seed set and which frameworks were seeded, for run metadata."""
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False

    try:
        import tensorflow  # noqa: F401

        tensorflow_available = True
    except ImportError:
        tensorflow_available = False

    return {
        "seed": _last_seed,
        "torch_available": torch_available,
        "tensorflow_available": tensorflow_available,
    }
