"""Classification evaluation: metrics, confusion matrix, and artifact persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.metrics import classification_report, confusion_matrix

if TYPE_CHECKING:
    from mlbag.tracking import RunManager


def evaluate_classifier(
    y_true: Any,
    y_pred: Any,
    *,
    labels: list[str] | None = None,
    run: "RunManager | None" = None,
    split_name: str = "test",
) -> dict:
    display_labels = labels if labels is not None else [str(c) for c in sorted(set(y_true) | set(y_pred))]
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if run is not None:
        import matplotlib.pyplot as plt
        import pandas as pd

        from mlbag.plotting.confusion import plot_confusion_matrix

        fig = plot_confusion_matrix(cm, display_labels, title=f"{split_name} confusion matrix")
        run.save_plot(fig, f"{split_name}_confusion_matrix")
        plt.close(fig)

        cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
        run.save_dataframe(cm_df.reset_index(names="true_label"), f"{split_name}_confusion_matrix")

        run.save_metrics(report, filename=f"{split_name}_metrics.json")

    return report
