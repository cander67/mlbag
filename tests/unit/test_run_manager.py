import json

import pytest
from sklearn.linear_model import LogisticRegression

from mlbag.config import MLBagConfig
from mlbag.tracking import RunManager


class DummyConfig(MLBagConfig):
    learning_rate: float = 1e-3


def test_init_creates_expected_subdirs(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="my_run")
    assert run.run_dir == tmp_path / "my_run"
    for sub in ("checkpoints", "plots", "metrics", "logs"):
        assert (run.run_dir / sub).is_dir()


def test_run_id_auto_generated_when_not_given(tmp_path):
    run = RunManager(base_dir=tmp_path, project="baseline")
    assert run.run_dir.exists()
    assert "baseline" in run.run_id


def test_resume_requires_existing_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunManager(base_dir=tmp_path, run_id="missing", resume=True)


def test_save_and_load_config_round_trip(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="cfg_run")
    config = DummyConfig(run_name="cfg_run", learning_rate=0.01)
    run.save_config(config)
    loaded = run.load_config(DummyConfig)
    assert loaded == config


def test_save_metrics_overwrite_and_append(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="metrics_run")
    run.save_metrics({"accuracy": 0.9}, filename="summary.json")
    assert run.load_metrics("summary.json") == {"accuracy": 0.9}

    run.save_metrics({"epoch": 1, "loss": 0.5}, filename="history.jsonl", append=True)
    run.save_metrics({"epoch": 2, "loss": 0.3}, filename="history.jsonl", append=True)
    history = run.load_metrics("history.jsonl")
    assert history == [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.3}]


def test_save_and_load_checkpoint_sklearn_estimator(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="ckpt_run")
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    path = run.save_checkpoint(model, name="model")
    assert path.suffix == ".joblib"

    loaded = run.load_checkpoint(None, path)
    assert loaded.predict([[1]])[0] == model.predict([[1]])[0]


def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="empty_run")
    assert run.latest_checkpoint() is None


def test_save_checkpoint_unsupported_type_raises(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="bad_run")
    with pytest.raises(TypeError):
        run.save_checkpoint(object(), name="model")


def test_sanitize_run_id_strips_unsafe_characters():
    assert RunManager.sanitize_run_id("my run / v2!") == "my_run_v2"


def test_mark_complete_writes_status(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="status_run")
    run.mark_complete("success")
    status = json.loads((run.run_dir / "status.json").read_text())
    assert status == {"status": "success"}


def test_list_runs_finds_completed_runs(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="listed_run")
    run.save_config(DummyConfig(run_name="listed_run"))
    run.mark_complete("success")

    summaries = RunManager.list_runs(tmp_path)
    assert len(summaries) == 1
    assert summaries[0].run_id == "listed_run"
    assert summaries[0].status == "success"
    assert summaries[0].config["run_name"] == "listed_run"


def test_list_runs_on_missing_base_dir_returns_empty(tmp_path):
    assert RunManager.list_runs(tmp_path / "does_not_exist") == []
