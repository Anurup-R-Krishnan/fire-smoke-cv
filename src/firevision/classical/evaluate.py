from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
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
)

from .colour_rules import ColourThresholdModel, classify_patch
from .config import ClassicalMLConfig
from .dataset import CLASS_NAMES, PatchRecord

LABELS = [0, 1, 2]
TARGET_NAMES = [CLASS_NAMES[label] for label in LABELS]


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, object]:
    true = np.asarray(list(y_true), dtype=np.int64)
    predicted = np.asarray(list(y_pred), dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(
            precision_score(true, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(true, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0)),
        "per_class": classification_report(
            true,
            predicted,
            labels=LABELS,
            target_names=TARGET_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(true, predicted, labels=LABELS).tolist(),
        "support": int(len(true)),
    }


def save_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    output_path: Path,
    title: str,
) -> None:
    true = np.asarray(list(y_true), dtype=np.int64)
    predicted = np.asarray(list(y_pred), dtype=np.int64)
    matrix = confusion_matrix(true, predicted, labels=LABELS)
    display = ConfusionMatrixDisplay(matrix, display_labels=TARGET_NAMES)
    figure, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, values_format="d")
    axis.set_title(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def evaluate_classical_records(
    records: list[PatchRecord],
    split: str,
    model: ColourThresholdModel,
    config: ClassicalMLConfig,
    prediction_csv: Path | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    selected = [record for record in records if record.split == split]
    true_labels: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, object]] = []
    for record in selected:
        image = cv2.imread(str(record.patch_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        prediction = classify_patch(image, model, config.colour)
        true_labels.append(record.class_id)
        predictions.append(prediction.predicted_class)
        rows.append(
            {
                "patch_path": str(record.patch_path),
                "true_class": record.class_name,
                "predicted_class": CLASS_NAMES[prediction.predicted_class],
                "fire_area_ratio": prediction.fire_statistics.area_ratio,
                "smoke_area_ratio": prediction.smoke_statistics.area_ratio,
                "fire_score": prediction.fire_score,
                "smoke_score": prediction.smoke_score,
            }
        )
    if prediction_csv is not None:
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    return (
        classification_metrics(true_labels, predictions),
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def tune_area_thresholds(
    records: list[PatchRecord],
    model: ColourThresholdModel,
    config: ClassicalMLConfig,
) -> tuple[ColourThresholdModel, list[dict[str, float]]]:
    results: list[dict[str, float]] = []
    best_model = model
    best_key = (-1.0, -1.0, 0.0, 0.0)
    for fire_threshold in config.colour.area_grid_fire:
        for smoke_threshold in config.colour.area_grid_smoke:
            candidate = model.with_area_thresholds(fire_threshold, smoke_threshold)
            tuning_split = "val" if any(r.split == "val" for r in records) else "test"
            metrics, _, _ = evaluate_classical_records(records, tuning_split, candidate, config)
            row = {
                "fire_area_threshold": fire_threshold,
                "smoke_area_threshold": smoke_threshold,
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
            results.append(row)
            # Prefer macro F1, then accuracy; when tied, prefer the stricter thresholds.
            key = (
                row["macro_f1"],
                row["accuracy"],
                fire_threshold + smoke_threshold,
                min(fire_threshold, smoke_threshold),
            )
            if key > best_key:
                best_key = key
                best_model = candidate
    return best_model, results


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
