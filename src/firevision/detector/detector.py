from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .config import DetectorTrainingConfig
from .runtime import resolve_device, set_global_seed


METRIC_KEYS = {
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
}


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements-detector.txt"
        ) from exc
    return YOLO


def _results_dict(metrics: Any) -> dict[str, float]:
    raw = getattr(metrics, "results_dict", {}) or {}
    return {str(key): float(value) for key, value in raw.items() if _is_finite_number(value)}


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalise_detector_metrics(metrics: Any) -> dict[str, float | None]:
    raw = _results_dict(metrics)
    output: dict[str, float | None] = {}
    for output_name, candidates in METRIC_KEYS.items():
        value = next((raw[key] for key in candidates if key in raw), None)
        output[output_name] = value
    precision = output["precision"]
    recall = output["recall"]
    output["f1"] = (
        None
        if precision is None or recall is None or precision + recall <= 0
        else 2.0 * precision * recall / (precision + recall)
    )
    return output


def select_best_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [trial for trial in trials if trial.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed detector trials")
    return max(
        completed,
        key=lambda trial: (
            float(trial["validation_metrics"].get("map50_95") or -1.0),
            float(trial["validation_metrics"].get("map50") or -1.0),
            float(trial["validation_metrics"].get("f1") or -1.0),
        ),
    )


def select_best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("f1") is not None]
    if not valid:
        raise ValueError("Threshold tuning produced no valid F1 values")
    return max(
        valid,
        key=lambda row: (
            float(row["f1"]),
            float(row.get("recall") or -1.0),
            float(row.get("precision") or -1.0),
            float(row["confidence"]),
        ),
    )


def _ultralytics_device(config: DetectorTrainingConfig) -> str | int:
    device = resolve_device(config.device)
    if device.type == "cuda":
        return device.index or 0
    return device.type


def train_detector_trials(config: DetectorTrainingConfig) -> dict[str, Any]:
    if not config.detection_dataset_yaml.exists():
        raise FileNotFoundError(
            f"Data Prep data.yaml not found: {config.detection_dataset_yaml}"
        )
    YOLO = _require_ultralytics()
    set_global_seed(config.seed)
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.output.report_dir.mkdir(parents=True, exist_ok=True)
    config.output.yolo_project_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, Any]] = []
    for image_size in config.detector.image_sizes:
        run_name = f"yolo11n_{image_size}"
        model = YOLO(config.detector.base_model)
        train_args: dict[str, Any] = {
            "data": str(config.detection_dataset_yaml),
            "imgsz": image_size,
            "epochs": config.detector.epochs,
            "batch": config.detector.batch_size,
            "workers": config.detector.workers,
            "patience": config.detector.patience,
            "amp": config.detector.amp,
            "cache": config.detector.cache,
            "optimizer": config.detector.optimizer,
            "weight_decay": config.detector.weight_decay,
            "close_mosaic": config.detector.close_mosaic,
            "device": _ultralytics_device(config),
            "project": str(config.output.yolo_project_dir),
            "name": run_name,
            "exist_ok": config.output.overwrite,
            "seed": config.seed,
            "deterministic": True,
            "plots": True,
            "verbose": True,
        }
        if config.detector.learning_rate is not None:
            train_args["lr0"] = config.detector.learning_rate
        model.train(**train_args)
        run_dir = config.output.yolo_project_dir / run_name
        checkpoint = run_dir / "weights" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Ultralytics did not create {checkpoint}")
        best_model = YOLO(str(checkpoint))
        validation = best_model.val(
            data=str(config.detection_dataset_yaml),
            split="val",
            imgsz=image_size,
            conf=0.001,
            iou=0.7,
            max_det=config.detector.max_det,
            device=_ultralytics_device(config),
            plots=True,
            verbose=False,
        )
        trials.append(
            {
                "status": "completed",
                "image_size": image_size,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "validation_metrics": normalise_detector_metrics(validation),
            }
        )

    selected = select_best_trial(trials)
    selected_checkpoint = Path(selected["checkpoint"])
    canonical_checkpoint = config.output.artifact_dir / "best_fire_smoke_detector.pt"
    shutil.copy2(selected_checkpoint, canonical_checkpoint)
    summary = {
        "selection_rule": "highest validation mAP50-95, then mAP50, then F1",
        "base_model": config.detector.base_model,
        "trials": trials,
        "selected_image_size": selected["image_size"],
        "selected_source_checkpoint": str(selected_checkpoint),
        "selected_checkpoint": str(canonical_checkpoint),
    }
    (config.output.artifact_dir / "detector_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (config.output.report_dir / "detector_size_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "image_size",
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95",
            "checkpoint",
            "selected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            metrics = trial["validation_metrics"]
            writer.writerow(
                {
                    "image_size": trial["image_size"],
                    **metrics,
                    "checkpoint": trial["checkpoint"],
                    "selected": trial["image_size"] == selected["image_size"],
                }
            )
    return summary


def tune_detector_thresholds(config: DetectorTrainingConfig) -> dict[str, Any]:
    YOLO = _require_ultralytics()
    summary_path = config.output.artifact_dir / "detector_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Train the detector before threshold tuning")
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = Path(training_summary["selected_checkpoint"])
    image_size = int(training_summary["selected_image_size"])
    model = YOLO(str(checkpoint))

    rows: list[dict[str, Any]] = []
    for confidence in config.detector.confidence_grid:
        for iou in config.detector.iou_grid:
            metrics = model.val(
                data=str(config.detection_dataset_yaml),
                split="val",
                imgsz=image_size,
                conf=confidence,
                iou=iou,
                max_det=config.detector.max_det,
                device=_ultralytics_device(config),
                plots=False,
                verbose=False,
            )
            row = {
                "confidence": confidence,
                "iou": iou,
                **normalise_detector_metrics(metrics),
            }
            rows.append(row)
    selected = select_best_threshold(rows)

    test_metrics_raw = model.val(
        data=str(config.detection_dataset_yaml),
        split="test",
        imgsz=image_size,
        conf=float(selected["confidence"]),
        iou=float(selected["iou"]),
        max_det=config.detector.max_det,
        device=_ultralytics_device(config),
        plots=True,
        verbose=False,
    )
    test_metrics = normalise_detector_metrics(test_metrics_raw)

    with (config.output.report_dir / "detector_threshold_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["confidence", "iou", "precision", "recall", "f1", "map50", "map50_95"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "selection_rule": "highest validation F1, then recall, precision, and stricter confidence",
        "selected_image_size": image_size,
        "selected_confidence": selected["confidence"],
        "selected_iou": selected["iou"],
        "selected_validation_metrics": {
            key: selected[key]
            for key in ("precision", "recall", "f1", "map50", "map50_95")
        },
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint),
        "test_evaluated_once_after_validation_selection": True,
    }
    (config.output.artifact_dir / "detector_thresholds.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
