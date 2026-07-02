import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")

from mlbag.tracking import RunManager  # noqa: E402
from mlbag.training.torch_loop import EarlyStopping, run_training  # noqa: E402


def _tiny_loaders():
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=5), DataLoader(dataset, batch_size=5)


def _tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def test_run_training_completes_and_writes_checkpoint(tmp_path):
    model = _tiny_model()
    train_loader, val_loader = _tiny_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    run = RunManager(base_dir=tmp_path, run_id="torch_smoke")

    history = run_training(
        model, train_loader, val_loader, optimizer, criterion, epochs=2, run=run, checkpoint_every=1
    )

    assert len(history["train_loss"]) == 2
    assert run.latest_checkpoint("model") is not None
    assert run.latest_checkpoint("best_model") is not None


def test_run_training_stops_early():
    model = _tiny_model()
    train_loader, val_loader = _tiny_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    history = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=20,
        early_stopping=EarlyStopping(patience=0, mode="min"),
    )
    # patience=0 always stops after the first epoch regardless of the loss trajectory
    assert len(history["train_loss"]) == 1
