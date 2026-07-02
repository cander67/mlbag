from mlbag.training.torch_loop import EarlyStopping


def test_early_stopping_triggers_after_patience_exceeded():
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(1.0) is False
    assert stopper.step(1.1) is False
    assert stopper.step(1.2) is True


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(1.0)
    stopper.step(1.1)
    stopper.step(0.5)
    assert stopper.step(0.6) is False
    assert stopper.step(0.7) is True


def test_early_stopping_mode_max():
    stopper = EarlyStopping(patience=1, mode="max")
    stopper.step(1.0)
    assert stopper.step(0.9) is True
