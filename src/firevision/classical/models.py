from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import ClassicalMLConfig
from .evaluate import classification_metrics
from .features import load_feature_archive


def _build_svm_model(config: ClassicalMLConfig, c_value: float, gamma: str | float) -> Pipeline:
    base_svm = SVC(
        C=c_value,
        gamma=gamma,
        kernel="rbf",
        class_weight="balanced",
        cache_size=config.svm.cache_size_mb,
        max_iter=config.svm.max_iter,
        decision_function_shape="ovr",
        random_state=config.seed,
    )
    if config.svm.probability:
        svm = CalibratedClassifierCV(base_svm, ensemble=False)
    else:
        svm = base_svm

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", svm),
        ]
    )


def _build_rf_model(config: ClassicalMLConfig, n_estimators: int, max_depth: int | None) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("rf", rf),
        ]
    )

def _build_et_model(config: ClassicalMLConfig, n_estimators: int, max_depth: int | None) -> Pipeline:
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=-1,
    )
    return Pipeline([("scale", StandardScaler()), ("et", et)])


def _build_xgb_model(config: ClassicalMLConfig, n_estimators: int, max_depth: int, learning_rate: float) -> Pipeline:
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=config.seed,
        n_jobs=-1,
    )
    return Pipeline([("scale", StandardScaler()), ("xgb", xgb)])


def _build_lgbm_model(config: ClassicalMLConfig, n_estimators: int, max_depth: int, learning_rate: float) -> Pipeline:
    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=config.seed,
        n_jobs=-1,
    )
    return Pipeline([("scale", StandardScaler()), ("lgbm", lgbm)])


def train_and_select_svm(config: ClassicalMLConfig, eval_split: str = "val") -> tuple[Pipeline, dict[str, object]]:
    train_path = config.output.artifact_dir / "features_train.npz"
    val_path = config.output.artifact_dir / f"features_{eval_split}.npz"
    X_train, y_train, _ = load_feature_archive(train_path)
    X_val, y_val, _ = load_feature_archive(val_path)

    trials: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)
    best_parameters: dict[str, object] = {}
    for c_value in config.ml_model.svm.c_values:
        for gamma in config.ml_model.svm.gamma_values:
            model = _build_svm_model(config, c_value, gamma)
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


def train_and_select_rf(config: ClassicalMLConfig, eval_split: str = "val") -> tuple[Pipeline, dict[str, object]]:
    train_path = config.output.artifact_dir / "features_train.npz"
    val_path = config.output.artifact_dir / f"features_{eval_split}.npz"
    X_train, y_train, _ = load_feature_archive(train_path)
    X_val, y_val, _ = load_feature_archive(val_path)

    trials: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)
    best_parameters: dict[str, object] = {}
    for n_estimators in config.ml_model.random_forest.n_estimators_values:
        for max_depth in config.ml_model.random_forest.max_depth_values:
            model = _build_rf_model(config, n_estimators, max_depth)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            metrics = classification_metrics(y_val, predictions)
            trial = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
            trials.append(trial)
            key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if key > best_key:
                best_key = key
                best_model = model
                best_parameters = {"n_estimators": n_estimators, "max_depth": max_depth}

    if best_model is None:
        raise RuntimeError("No Random Forest model was trained")
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output.artifact_dir / "rf_model.joblib"
    joblib.dump(best_model, model_path)

    trials_path = config.output.report_dir / "rf_validation_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n_estimators", "max_depth", "macro_f1", "accuracy"])
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "model_path": str(model_path),
        "best_parameters": best_parameters,
        "best_validation_macro_f1": best_key[0],
        "best_validation_accuracy": best_key[1],
        "trials": trials,
    }
    (config.output.artifact_dir / "rf_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_model, summary


def train_and_select_extra_trees(config: ClassicalMLConfig, eval_split: str = "val") -> tuple[Pipeline, dict[str, object]]:
    train_path = config.output.artifact_dir / "features_train.npz"
    val_path = config.output.artifact_dir / f"features_{eval_split}.npz"
    X_train, y_train, _ = load_feature_archive(train_path)
    X_val, y_val, _ = load_feature_archive(val_path)

    trials: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)
    best_parameters: dict[str, object] = {}
    for n_estimators in config.ml_model.extra_trees.n_estimators_values:
        for max_depth in config.ml_model.extra_trees.max_depth_values:
            model = _build_et_model(config, n_estimators, max_depth)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            metrics = classification_metrics(y_val, predictions)
            trial = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
            trials.append(trial)
            key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if key > best_key:
                best_key = key
                best_model = model
                best_parameters = {"n_estimators": n_estimators, "max_depth": max_depth}

    if best_model is None:
        raise RuntimeError("No Extra Trees model was trained")
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output.artifact_dir / "et_model.joblib"
    joblib.dump(best_model, model_path)

    trials_path = config.output.report_dir / "et_validation_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n_estimators", "max_depth", "macro_f1", "accuracy"])
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "model_path": str(model_path),
        "best_parameters": best_parameters,
        "best_validation_macro_f1": best_key[0],
        "best_validation_accuracy": best_key[1],
        "trials": trials,
    }
    (config.output.artifact_dir / "et_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_model, summary


def train_and_select_xgboost(config: ClassicalMLConfig, eval_split: str = "val") -> tuple[Pipeline, dict[str, object]]:
    train_path = config.output.artifact_dir / "features_train.npz"
    val_path = config.output.artifact_dir / f"features_{eval_split}.npz"
    X_train, y_train, _ = load_feature_archive(train_path)
    X_val, y_val, _ = load_feature_archive(val_path)

    trials: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)
    best_parameters: dict[str, object] = {}
    for n_estimators in config.ml_model.xgboost.n_estimators_values:
        for max_depth in config.ml_model.xgboost.max_depth_values:
            for lr in config.ml_model.xgboost.learning_rate_values:
                model = _build_xgb_model(config, n_estimators, max_depth, lr)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                metrics = classification_metrics(y_val, predictions)
                trial = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                }
                trials.append(trial)
                key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
                if key > best_key:
                    best_key = key
                    best_model = model
                    best_parameters = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": lr}

    if best_model is None:
        raise RuntimeError("No XGBoost model was trained")
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output.artifact_dir / "xgb_model.joblib"
    joblib.dump(best_model, model_path)

    trials_path = config.output.report_dir / "xgb_validation_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n_estimators", "max_depth", "learning_rate", "macro_f1", "accuracy"])
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "model_path": str(model_path),
        "best_parameters": best_parameters,
        "best_validation_macro_f1": best_key[0],
        "best_validation_accuracy": best_key[1],
        "trials": trials,
    }
    (config.output.artifact_dir / "xgb_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_model, summary


def train_and_select_lightgbm(config: ClassicalMLConfig, eval_split: str = "val") -> tuple[Pipeline, dict[str, object]]:
    train_path = config.output.artifact_dir / "features_train.npz"
    val_path = config.output.artifact_dir / f"features_{eval_split}.npz"
    X_train, y_train, _ = load_feature_archive(train_path)
    X_val, y_val, _ = load_feature_archive(val_path)

    trials: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_key = (-1.0, -1.0)
    best_parameters: dict[str, object] = {}
    for n_estimators in config.ml_model.lightgbm.n_estimators_values:
        for max_depth in config.ml_model.lightgbm.max_depth_values:
            for lr in config.ml_model.lightgbm.learning_rate_values:
                model = _build_lgbm_model(config, n_estimators, max_depth, lr)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                metrics = classification_metrics(y_val, predictions)
                trial = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                }
                trials.append(trial)
                key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
                if key > best_key:
                    best_key = key
                    best_model = model
                    best_parameters = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": lr}

    if best_model is None:
        raise RuntimeError("No LightGBM model was trained")
    config.output.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output.artifact_dir / "lgbm_model.joblib"
    joblib.dump(best_model, model_path)

    trials_path = config.output.report_dir / "lgbm_validation_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n_estimators", "max_depth", "learning_rate", "macro_f1", "accuracy"])
        writer.writeheader()
        writer.writerows(trials)

    summary = {
        "model_path": str(model_path),
        "best_parameters": best_parameters,
        "best_validation_macro_f1": best_key[0],
        "best_validation_accuracy": best_key[1],
        "trials": trials,
    }
    (config.output.artifact_dir / "lgbm_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return best_model, summary


def evaluate_model(
    model: Pipeline,
    config: ClassicalMLConfig,
    split: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    feature_path = config.output.artifact_dir / f"features_{split}.npz"
    X, y, paths = load_feature_archive(feature_path)
    predictions = model.predict(X)
    metrics = classification_metrics(y, predictions)
    return metrics, y, predictions, paths
