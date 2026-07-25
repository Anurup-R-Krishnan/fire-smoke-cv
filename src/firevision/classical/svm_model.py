from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ClassicalMLConfig
from .evaluate import classification_metrics
from .features import load_feature_archive


def _build_model(config: ClassicalMLConfig, c_value: float, gamma: str | float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    C=c_value,
                    gamma=gamma,
                    kernel="rbf",
                    class_weight="balanced",
                    probability=config.svm.probability,
                    cache_size=config.svm.cache_size_mb,
                    max_iter=config.svm.max_iter,
                    decision_function_shape="ovr",
                    random_state=config.seed,
                ),
            ),
        ]
    )


def train_and_select_svm(config: ClassicalMLConfig) -> tuple[Pipeline, dict[str, object]]:
    train_path = config.output.artifact_dir / "features_train.npz"
    val_path = config.output.artifact_dir / "features_val.npz"
    X_train, y_train, _ = load_feature_archive(train_path)
    X_val, y_val, _ = load_feature_archive(val_path)

    trials: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)
    best_parameters: dict[str, object] = {}
    for c_value in config.svm.c_values:
        for gamma in config.svm.gamma_values:
            model = _build_model(config, c_value, gamma)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            metrics = classification_metrics(y_val, predictions)
            trial = {
                "C": c_value,
                "gamma": gamma,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
            trials.append(trial)
            key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if key > best_key:
                best_key = key
                best_model = model
                best_parameters = {"C": c_value, "gamma": gamma}

    if best_model is None:
        raise RuntimeError("No SVM model was trained")
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output.artifact_dir / "hog_lbp_colour_rbf_svm.joblib"
    joblib.dump(best_model, model_path)

    trials_path = config.output.report_dir / "svm_validation_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["C", "gamma", "macro_f1", "accuracy"])
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "model_path": str(model_path),
        "best_parameters": best_parameters,
        "best_validation_macro_f1": best_key[0],
        "best_validation_accuracy": best_key[1],
        "trials": trials,
    }
    (config.output.artifact_dir / "svm_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_model, summary


def evaluate_svm(
    model: Pipeline,
    config: ClassicalMLConfig,
    split: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    feature_path = config.output.artifact_dir / f"features_{split}.npz"
    X, y, paths = load_feature_archive(feature_path)
    predictions = model.predict(X)
    metrics = classification_metrics(y, predictions)
    return metrics, y, predictions, paths
