import random

import numpy as np

from mlbag.seeding import get_seed_state, seed_everything


def test_seed_everything_reproducible_random():
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_seed_everything_reproducible_numpy():
    seed_everything(7)
    first = np.random.rand(5)
    seed_everything(7)
    second = np.random.rand(5)
    assert (first == second).all()


def test_different_seeds_diverge():
    seed_everything(1)
    a = random.random()
    seed_everything(2)
    b = random.random()
    assert a != b


def test_get_seed_state_reports_last_seed():
    seed_everything(99)
    state = get_seed_state()
    assert state["seed"] == 99
