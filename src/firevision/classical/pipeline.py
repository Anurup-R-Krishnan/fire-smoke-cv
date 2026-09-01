from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .colour_rules import fit_pixel_thresholds
from .config import ClassicalMLConfig
from .dataset import CLASS_NAMES, PatchRecord, load_patch_manifest, prepare_patch_dataset
from .evaluate import (
    evaluate_classical_records,
    save_confusion_matrix,
    save_metrics,
    tune_area_thresholds,
)
from .features import extract_split_features
from .models import (
    evaluate_model,
    train_and_select_svm,
    train_and_select_rf,
    train_and_select_extra_trees,
    train_and_select_xgboost,
    train_and_select_lightgbm,
)
from .plotting import generate_comparison_charts


def _clean_outputs(config: ClassicalMLConfig) -> None:
    if not config.output.overwrite:
        return
    for directory in (config.output.report_dir, config.output.artifact_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def _write_threshold_trials(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fire_area_threshold", "smoke_area_threshold", "macro_f1", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_prediction_csv(
    paths: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["patch_path", "true_class", "predicted_class"]
        )
        writer.writeheader()
        for patch_path, true_id, predicted_id in zip(paths, true, predicted, strict=True):
            writer.writerow(
                {
                    "patch_path": str(patch_path),
                    "true_class": CLASS_NAMES[int(true_id)],
                    "predicted_class": CLASS_NAMES[int(predicted_id)],
                }
            )


def _failure_gallery(
    paths: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
    maximum: int = 12,
) -> None:
    failures = np.flatnonzero(true != predicted)[:maximum]
    if not len(failures):
        return
    columns = 4
    rows = int(np.ceil(len(failures) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis in axes_array:
        axis.axis("off")
    for axis, index in zip(axes_array, failures, strict=False):
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"true={CLASS_NAMES[int(true[index])]}\npred={CLASS_NAMES[int(predicted[index])]}",
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _patch_counts(records: list[PatchRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counter = Counter(record.class_name for record in records if record.split == split)
        counts[split] = {name: int(counter.get(name, 0)) for name in CLASS_NAMES.values()}
    return counts


def _comparison_rows(
    classical_val: dict[str, object] | None,
    classical_test: dict[str, object],
    ml_metrics_list: list[tuple[str, dict[str, object] | None, dict[str, object]]],
) -> list[dict[str, object]]:
    rows = []
    if classical_val is not None:
        rows.append(
            {
                "method": "colour+morphology",
                "split": "val",
                "accuracy": classical_val["accuracy"],
                "macro_precision": classical_val["macro_precision"],
                "macro_recall": classical_val["macro_recall"],
                "macro_f1": classical_val["macro_f1"],
                "support": classical_val["support"],
            }
        )
    rows.append(
        {
            "method": "colour+morphology",
            "split": "test",
            "accuracy": classical_test["accuracy"],
            "macro_precision": classical_test["macro_precision"],
            "macro_recall": classical_test["macro_recall"],
            "macro_f1": classical_test["macro_f1"],
            "support": classical_test["support"],
        }
    )
    for ml_name, ml_val, ml_test in ml_metrics_list:
        if ml_val is not None:
            rows.append(
                {
                    "method": ml_name,
                    "split": "val",
                    "accuracy": ml_val["accuracy"],
                    "macro_precision": ml_val["macro_precision"],
                    "macro_recall": ml_val["macro_recall"],
                    "macro_f1": ml_val["macro_f1"],
                    "support": ml_val["support"],
                }
            )
        rows.append(
            {
                "method": ml_name,
                "split": "test",
                "accuracy": ml_test["accuracy"],
                "macro_precision": ml_test["macro_precision"],
                "macro_recall": ml_test["macro_recall"],
                "macro_f1": ml_test["macro_f1"],
                "support": ml_test["support"],
            }
        )
    return rows


def _write_report(
    config: ClassicalMLConfig,
    summary: dict[str, object],
    comparison: list[dict[str, object]],
) -> None:
    counts = summary["patch_counts"]
    thresholds = summary["colour_thresholds"]
    has_val = sum(counts.get("val", {}).values()) > 0
    lines = [
        "# Classical ML Report: Classical CV and Conventional ML",
        "",
        "## Leakage controls",
        "",
        "- Patches inherit the original Data Prep split.",
        "- Pixel colour thresholds are fitted using training patches only.",
    ]
    if has_val:
        lines.append("- Mask-area thresholds are selected using validation patches only.")
    else:
        lines.append("- Note: Dataset has no validation split (val=0); evaluation is reported on held-out test patches.")
    lines.extend(
        [
            "- Final metrics are reported on untouched test patches.",
            "",
            "## Patch dataset",
            "",
            "| Split | Fire | Smoke | Normal |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in ("train", "val", "test"):
        values = counts[split]
        lines.append(
            f"| {split} | {values['fire']} | {values['smoke']} | {values['normal']} |"
        )
    lines.extend(
        [
            "",
            "## Selected classical thresholds",
            "",
            f"- Fire mask-area threshold: `{thresholds['fire_area_threshold']:.5f}`",
            f"- Smoke mask-area threshold: `{thresholds['smoke_area_threshold']:.5f}`",
            f"- Fire pixel vote requirement: `{thresholds['fire_vote_threshold']}` of 7 rules",
            f"- Smoke pixel vote requirement: `{thresholds['smoke_vote_threshold']}` of 7 rules",
            "",
            "## Results",
            "",
            "| Method | Split | Accuracy | Macro precision | Macro recall | Macro F1 | Samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            "| {method} | {split} | {accuracy:.4f} | {macro_precision:.4f} | "
            "{macro_recall:.4f} | {macro_f1:.4f} | {support} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `colour_thresholds.json`: fitted pixel and area thresholds",
            "- `*_model.joblib`: trained ML pipelines",
            "- `features_{train,val,test}.npz`: deterministic feature archives",
            "- Confusion matrices and failure galleries in `reports/classical/`",
            "",
            "## Interpretation requirement",
            "",
            "Do not present the classical system as the final detector. Document failures on sunsets, lamps, clouds, steam, reflections, and low-contrast smoke. Detector Training must test whether learned models improve these weaknesses.",
        ]
    )
    (config.output.report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config: ClassicalMLConfig) -> dict[str, object]:
    if not (config.dataset_dir / "images" / "train").exists():
        raise FileNotFoundError(
            f"Processed Data Prep dataset not found at {config.dataset_dir}. Run Data Prep first."
        )
    _clean_outputs(config)
    records = prepare_patch_dataset(config)
    counts = _patch_counts(records)
    has_val = any(r.split == "val" for r in records)
    eval_split = "val" if has_val else "test"
    for split, split_counts in counts.items():
        total = sum(split_counts.values())
        if total == 0:
            continue  # Split not present in this dataset (e.g. no val split)
        missing = [name for name, count in split_counts.items() if count == 0]
        if missing:
            raise ValueError(f"Split {split} has no patches for classes: {', '.join(missing)}")

    initial_thresholds = fit_pixel_thresholds(records, config.colour, config.seed)
    tuned_thresholds, threshold_trials = tune_area_thresholds(records, initial_thresholds, config)
    threshold_path = config.output.artifact_dir / "colour_thresholds.json"
    tuned_thresholds.save(threshold_path)
    _write_threshold_trials(
        threshold_trials, config.output.report_dir / "classical_threshold_trials.csv"
    )

    if has_val:
        classical_val, classical_val_true, classical_val_pred = evaluate_classical_records(
            records,
            "val",
            tuned_thresholds,
            config,
            config.output.report_dir / "classical_val_predictions.csv",
        )
        save_metrics(classical_val, config.output.report_dir / "classical_val_metrics.json")
    else:
        classical_val = None

    classical_test, classical_test_true, classical_test_pred = evaluate_classical_records(
        records,
        "test",
        tuned_thresholds,
        config,
        config.output.report_dir / "classical_test_predictions.csv",
    )
    save_metrics(classical_test, config.output.report_dir / "classical_test_metrics.json")
    save_confusion_matrix(
        classical_test_true,
        classical_test_pred,
        config.output.report_dir / "classical_test_confusion_matrix.png",
        "Classical colour/morphology baseline — test",
    )
    classical_test_records = [record for record in records if record.split == "test"]
    _failure_gallery(
        np.asarray([str(record.patch_path) for record in classical_test_records]),
        classical_test_true,
        classical_test_pred,
        config.output.report_dir / "classical_failure_gallery.png",
        "Classical baseline failures",
    )

    for split in ("train", "val", "test") if has_val else ("train", "test"):
        extract_split_features(records, split, config)
    
    if config.ml_model.train_all:
        models_to_train = [
            ("svm", "SVM", train_and_select_svm),
            ("rf", "Random Forest", train_and_select_rf),
            ("et", "Extra Trees", train_and_select_extra_trees),
            ("xgb", "XGBoost", train_and_select_xgboost),
            ("lgbm", "LightGBM", train_and_select_lightgbm),
        ]
    else:
        model_type = config.ml_model.type
        if model_type == "random_forest":
            models_to_train = [("rf", "Random Forest", train_and_select_rf)]
        else:
            models_to_train = [("svm", "SVM", train_and_select_svm)]

    ml_metrics_list = []
    ml_trainings = {}
    plot_results = []
    
    for short_name, ml_name, train_func in models_to_train:
        ml_model, ml_training = train_func(config, "val" if has_val else "test")
        ml_trainings[short_name] = ml_training
        
        if has_val:
            ml_val, ml_val_true, ml_val_pred, ml_val_paths, ml_val_proba = evaluate_model(ml_model, config, "val")
            save_metrics(ml_val, config.output.report_dir / f"{short_name}_val_metrics.json")
        else:
            ml_val = None

        ml_test, ml_test_true, ml_test_pred, ml_test_paths, ml_test_proba = evaluate_model(ml_model, config, "test")
        ml_results[short_name] = {"validation": ml_val, "test": ml_test}
        
        ml_metrics_list.append((ml_name, ml_val, ml_test))
        # Store for plotting later
        plot_results.append({
            "name": ml_name,
            "y_true": ml_test_true,
            "y_pred": ml_test_pred,
            "y_proba": ml_test_proba,
        })
        
        save_metrics(ml_test, config.output.report_dir / f"{short_name}_test_metrics.json")
        save_confusion_matrix(
            ml_test_true,
            ml_test_pred,
            config.output.report_dir / f"{short_name}_test_confusion_matrix.png",
            f"{ml_name} — test",
        )
        _write_prediction_csv(
            ml_test_paths,
            ml_test_true,
            ml_test_pred,
            config.output.report_dir / f"{short_name}_test_predictions.csv",
        )
        _failure_gallery(
            ml_test_paths,
            ml_test_true,
            ml_test_pred,
            config.output.report_dir / f"{short_name}_failure_gallery.png",
            f"{ml_name} failures",
        )

    comparison = _comparison_rows(classical_val, classical_test, ml_metrics_list)
    comparison_path = config.output.report_dir / "method_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    summary: dict[str, object] = {
        "patch_counts": counts,
        "colour_thresholds": tuned_thresholds.to_dict(),
        "classical_validation": classical_val,
        "classical_test": classical_test,
        "ml_trainings": ml_trainings,
        "ml_results": ml_results,
        "threshold_model": str(threshold_path),
    }
    summary.update({f"{name}_test": result["test"] for name, result in ml_results.items()})
    (config.output.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(config, summary, comparison)
    
    if "plot_results" in locals():
        generate_comparison_charts(config.output.report_dir, comparison_path, plot_results)
        
    return summary
