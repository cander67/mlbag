from mlbag.evaluation import evaluate_classifier, evaluate_regressor
from mlbag.tracking import RunManager


def test_evaluate_classifier_known_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    report = evaluate_classifier(y_true, y_pred, labels=["neg", "pos"])
    assert report["accuracy"] == 0.75


def test_evaluate_classifier_persists_artifacts_when_run_given(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="eval_run")
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    evaluate_classifier(y_true, y_pred, labels=["neg", "pos"], run=run, split_name="val")

    assert (run.plots_dir / "val_confusion_matrix.png").exists()
    assert (run.metrics_dir / "val_confusion_matrix.csv").exists()
    assert (run.metrics_dir / "val_metrics.json").exists()


def test_evaluate_regressor_known_values():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    metrics = evaluate_regressor(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_evaluate_regressor_persists_artifacts_when_run_given(tmp_path):
    run = RunManager(base_dir=tmp_path, run_id="reg_run")
    evaluate_regressor([1.0, 2.0], [1.5, 2.5], run=run, split_name="test")
    assert (run.plots_dir / "test_residuals.png").exists()
    assert (run.metrics_dir / "test_metrics.json").exists()
