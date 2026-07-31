from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .classifier_train import train_enabled_classifiers
from .config import DetectorTrainingConfig
from .detector import train_detector_trials, tune_detector_thresholds
from .gradcam import generate_failure_gallery


def _validate_inputs(config: DetectorTrainingConfig, run_classifiers: bool, run_detector: bool) -> None:
    if run_classifiers:
        for split in ("train", "val", "test"):
            split_dir = config.classification_patch_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Classical ML patch dataset missing: {split_dir}. Run Classical ML first."
                )
    if run_detector and not config.detection_dataset_yaml.exists():
        raise FileNotFoundError(
            f"Data Prep detection data.yaml missing: {config.detection_dataset_yaml}"
        )


def _write_report(config: DetectorTrainingConfig, summary: dict[str, Any]) -> None:
    classifier_rows = [
        row for row in summary.get("classifiers", []) if row.get("status") == "completed"
    ]
    lines = [
        "# Detector Training Report: Deep Learning",
        "",
        "## Selection discipline",
        "",
        "- Classifier architecture selection uses validation macro F1.",
        "- Detector input-size selection uses validation mAP50-95.",
        "- Confidence and NMS IoU are tuned on validation data.",
        "- The detector test split is evaluated only after thresholds are fixed.",
        "",
        "## Classifiers",
        "",
    ]
    if classifier_rows:
        lines.extend(
            [
                "| Model | Best validation macro F1 | Test macro F1 | Checkpoint |",
                "|---|---:|---:|---|",
            ]
        )
        for row in classifier_rows:
            lines.append(
                f"| {row['model']} | {float(row['best_validation_macro_f1']):.4f} | "
                f"{float(row['test_metrics']['macro_f1']):.4f} | `{row['checkpoint']}` |"
            )
    else:
        lines.append("Classifier training was skipped.")

    lines.extend(["", "## Detector", ""])
    detector = summary.get("detector_training")
    thresholds = summary.get("detector_thresholds")
    if detector:
        lines.append(
            f"Selected input size: **{detector['selected_image_size']}** using validation mAP50-95."
        )
        lines.append(f"Checkpoint: `{detector['selected_checkpoint']}`")
    else:
        lines.append("Detector training was skipped.")
    if thresholds:
        lines.extend(
            [
                "",
                f"Selected confidence threshold: **{float(thresholds['selected_confidence']):.3f}**",
                f"Selected NMS IoU threshold: **{float(thresholds['selected_iou']):.3f}**",
                f"Final test F1: **{float(thresholds['test_metrics']['f1'] or 0):.4f}**",
                f"Final test mAP50-95: **{float(thresholds['test_metrics']['map50_95'] or 0):.4f}**",
            ]
        )
    lines.extend(
        [
            "",
            "## Required interpretation",
            "",
            "Inspect false positives on sunsets, lamps, reflections, steam, clouds, and screens showing fire. Grad-CAM is explanatory evidence for the classifier only; it does not prove causal reasoning. The detector checkpoint becomes the frame-level input to Video Fusion temporal fusion.",
        ]
    )
    config.output.report_dir.mkdir(parents=True, exist_ok=True)
    (config.output.report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    config: DetectorTrainingConfig,
    *,
    run_classifiers: bool = True,
    run_detector: bool = True,
    run_threshold_tuning: bool = True,
    run_gradcam: bool = True,
) -> dict[str, Any]:
    _validate_inputs(config, run_classifiers, run_detector)
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.output.report_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    if run_classifiers:
        classifier_results = train_enabled_classifiers(config)
        summary["classifiers"] = classifier_results
        if run_gradcam:
            gradcam_outputs: dict[str, list[str]] = {}
            for row in classifier_results:
                if row.get("status") != "completed":
                    continue
                model_name = str(row["model"])
                predictions = config.output.report_dir / model_name / "test_predictions.csv"
                outputs = generate_failure_gallery(
                    config,
                    Path(str(row["checkpoint"])),
                    predictions,
                    maximum_images=16,
                )
                gradcam_outputs[model_name] = [str(path) for path in outputs]
            summary["gradcam_outputs"] = gradcam_outputs
    if run_detector:
        summary["detector_training"] = train_detector_trials(config)
        if run_threshold_tuning:
            summary["detector_thresholds"] = tune_detector_thresholds(config)

    (config.output.artifact_dir / "detector_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(config, summary)
    return summary
