from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probabilities: np.ndarray | None,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    true = np.asarray(list(y_true), dtype=np.int64)
    predicted = np.asarray(list(y_pred), dtype=np.int64)
    labels = list(range(len(class_names)))
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(
            precision_score(true, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(true, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(
            true,
            predicted,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
        "support": int(len(true)),
    }
    if probabilities is not None and len(np.unique(true)) == len(class_names):
        try:
            result["macro_roc_auc_ovr"] = float(
                roc_auc_score(true, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
        except ValueError:
            result["macro_roc_auc_ovr"] = None
    else:
        result["macro_roc_auc_ovr"] = None
    return result


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=list(class_names))
    figure, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, values_format="d")
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_training_curves(history: list[dict[str, float | int | str]], path: Path, title: str) -> None:
    if not history:
        return
    epochs = [int(row["global_epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    val_f1 = [float(row["val_macro_f1"]) for row in history]

    figure, first_axis = plt.subplots(figsize=(8, 5))
    first_axis.plot(epochs, train_loss, label="train loss")
    first_axis.plot(epochs, val_loss, label="validation loss")
    first_axis.set_xlabel("Epoch")
    first_axis.set_ylabel("Loss")
    second_axis = first_axis.twinx()
    second_axis.plot(epochs, val_f1, linestyle="--", label="validation macro F1")
    second_axis.set_ylabel("Macro F1")
    lines = first_axis.get_lines() + second_axis.get_lines()
    first_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    first_axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_json(data: dict[str, object] | list[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_prediction_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
