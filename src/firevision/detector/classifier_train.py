from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .classifier_data import balanced_class_weights, build_loaders
from .classifier_models import (
    build_classifier,
    freeze_backbone,
    parameter_counts,
    trainable_parameters,
    unfreeze_last_feature_blocks,
)
from .config import ModelConfig, DetectorTrainingConfig
from .metrics import (
    classification_metrics,
    save_confusion_matrix,
    save_json,
    save_prediction_rows,
    save_training_curves,
)
from .runtime import resolve_device, set_global_seed


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []
    probability_rows: list[np.ndarray] = []
    paths: list[str] = []

    for tensors, labels, batch_paths in loader:
        tensors = tensors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with _autocast(device, amp):
                logits = model(tensors)
                loss = criterion(logits, labels)
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        probabilities = torch.softmax(logits.detach(), dim=1)
        predicted = probabilities.argmax(dim=1)
        batch_size = labels.shape[0]
        running_loss += float(loss.detach().cpu()) * batch_size
        true_labels.extend(labels.detach().cpu().tolist())
        predictions.extend(predicted.detach().cpu().tolist())
        probability_rows.append(probabilities.cpu().numpy())
        paths.extend(list(batch_paths))

    probabilities_np = (
        np.concatenate(probability_rows, axis=0)
        if probability_rows
        else np.empty((0, 0), dtype=np.float32)
    )
    return (
        running_loss / max(1, len(true_labels)),
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        probabilities_np,
        paths,
    )


def _train_stage(
    *,
    stage_name: str,
    model: nn.Module,
    model_name: str,
    epochs: int,
    start_global_epoch: int,
    learning_rate: float,
    loaders,
    criterion: nn.Module,
    device: torch.device,
    config: DetectorTrainingConfig,
    checkpoint_path: Path,
    history: list[dict[str, float | int | str]],
    best_state: dict[str, torch.Tensor] | None,
    best_f1: float,
) -> tuple[int, dict[str, torch.Tensor] | None, float, int]:
    if epochs <= 0:
        return start_global_epoch, best_state, best_f1, 0
    optimizer = AdamW(
        trainable_parameters(model),
        lr=learning_rate,
        weight_decay=config.classifier.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    amp_enabled = config.classifier.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    without_improvement = 0
    epochs_run = 0

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_global_epoch + local_epoch
        started = time.perf_counter()
        train_loss, _, _, _, _ = _run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler, amp_enabled
        )
        with torch.no_grad():
            val_loss, y_true, y_pred, probabilities, _ = _run_epoch(
                model, loaders["val"], criterion, device, None, None, amp_enabled
            )
        metrics = classification_metrics(
            y_true, y_pred, probabilities, config.classifier_classes
        )
        macro_f1 = float(metrics["macro_f1"])
        scheduler.step(macro_f1)
        history.append(
            {
                "stage": stage_name,
                "global_epoch": global_epoch,
                "local_epoch": local_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": float(metrics["accuracy"]),
                "val_macro_f1": macro_f1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.perf_counter() - started,
            }
        )
        epochs_run += 1
        if macro_f1 > best_f1 + 1e-6:
            best_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": model_name,
                    "state_dict": best_state,
                    "class_names": config.classifier_classes,
                    "image_size": config.classifier.image_size,
                    "validation_macro_f1": best_f1,
                    "global_epoch": global_epoch,
                },
                checkpoint_path,
            )
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= config.classifier.patience:
                break
    return start_global_epoch + epochs_run, best_state, best_f1, epochs_run


def evaluate_classifier(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    config: DetectorTrainingConfig,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    with torch.no_grad():
        loss, y_true, y_pred, probabilities, paths = _run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            amp=config.classifier.amp,
        )
    metrics = classification_metrics(y_true, y_pred, probabilities, config.classifier_classes)
    metrics["loss"] = loss
    return metrics, y_true, y_pred, probabilities, paths, loss


def train_classifier(config: DetectorTrainingConfig, model_name: str) -> dict[str, object]:
    if model_name not in config.classifier.models:
        raise KeyError(f"No configuration for classifier {model_name}")
    model_config: ModelConfig = config.classifier.models[model_name]
    if not model_config.enabled:
        return {"model": model_name, "status": "disabled"}

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    datasets, loaders = build_loaders(config, model_config.batch_size, device)
    class_weights = balanced_class_weights(
        datasets["train"], len(config.classifier_classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.classifier.label_smoothing,
    )

    bundle = build_classifier(
        model_name,
        class_count=len(config.classifier_classes),
        pretrained=model_config.pretrained,
    )
    model = bundle.model.to(device)
    freeze_backbone(model_name, model)
    initial_counts = parameter_counts(model)

    checkpoint_path = config.output.artifact_dir / f"{model_name}_best.pt"
    history: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    global_epoch = 0

    global_epoch, best_state, best_f1, _ = _train_stage(
        stage_name="head",
        model=model,
        model_name=model_name,
        epochs=model_config.stage1_epochs,
        start_global_epoch=global_epoch,
        learning_rate=model_config.head_learning_rate,
        loaders=loaders,
        criterion=criterion,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        history=history,
        best_state=best_state,
        best_f1=best_f1,
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    unfreeze_last_feature_blocks(
        model_name, model, model_config.unfreeze_last_blocks
    )
    finetune_counts = parameter_counts(model)
    global_epoch, best_state, best_f1, _ = _train_stage(
        stage_name="finetune",
        model=model,
        model_name=model_name,
        epochs=model_config.stage2_epochs,
        start_global_epoch=global_epoch,
        learning_rate=model_config.finetune_learning_rate,
        loaders=loaders,
        criterion=criterion,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        history=history,
        best_state=best_state,
        best_f1=best_f1,
    )

    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation, _, _, _, _, _ = evaluate_classifier(
        model, loaders["val"], criterion, device, config
    )
    test, y_true, y_pred, probabilities, paths, _ = evaluate_classifier(
        model, loaders["test"], criterion, device, config
    )

    model_report_dir = config.output.report_dir / model_name
    model_report_dir.mkdir(parents=True, exist_ok=True)
    save_json(validation, model_report_dir / "validation_metrics.json")
    save_json(test, model_report_dir / "test_metrics.json")
    save_confusion_matrix(
        y_true,
        y_pred,
        config.classifier_classes,
        model_report_dir / "test_confusion_matrix.png",
        f"{model_name} test confusion matrix",
    )
    save_training_curves(
        history,
        model_report_dir / "training_curves.png",
        f"{model_name} training",
    )
    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        row: dict[str, object] = {
            "path": path,
            "true_class": config.classifier_classes[int(y_true[index])],
            "predicted_class": config.classifier_classes[int(y_pred[index])],
            "correct": bool(y_true[index] == y_pred[index]),
        }
        for class_id, class_name in enumerate(config.classifier_classes):
            row[f"probability_{class_name}"] = float(probabilities[index, class_id])
        rows.append(row)
    save_prediction_rows(rows, model_report_dir / "test_predictions.csv")

    history_path = model_report_dir / "training_history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    summary: dict[str, object] = {
        "model": model_name,
        "status": "completed",
        "device": str(device),
        "configuration": asdict(model_config),
        "checkpoint": str(checkpoint_path),
        "class_weights": class_weights.detach().cpu().tolist(),
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "frozen_stage_parameters": initial_counts,
        "finetune_stage_parameters": finetune_counts,
        "best_validation_macro_f1": best_f1,
        "validation_metrics": validation,
        "test_metrics": test,
    }
    save_json(summary, model_report_dir / "summary.json")
    return summary


def train_enabled_classifiers(config: DetectorTrainingConfig) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for model_name in ("mobilenet_v3_small", "vgg16"):
        if model_name in config.classifier.models:
            results.append(train_classifier(config, model_name))
    completed = [row for row in results if row.get("status") == "completed"]
    if completed:
        best = max(
            completed,
            key=lambda row: float(row["best_validation_macro_f1"]),
        )
        comparison_path = config.output.report_dir / "classifier_comparison.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "model",
                "status",
                "best_validation_macro_f1",
                "test_accuracy",
                "test_macro_precision",
                "test_macro_recall",
                "test_macro_f1",
                "checkpoint",
                "selected_by_validation",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                if row.get("status") != "completed":
                    writer.writerow({"model": row["model"], "status": row["status"]})
                    continue
                test = row["test_metrics"]
                writer.writerow(
                    {
                        "model": row["model"],
                        "status": row["status"],
                        "best_validation_macro_f1": row["best_validation_macro_f1"],
                        "test_accuracy": test["accuracy"],
                        "test_macro_precision": test["macro_precision"],
                        "test_macro_recall": test["macro_recall"],
                        "test_macro_f1": test["macro_f1"],
                        "checkpoint": row["checkpoint"],
                        "selected_by_validation": row["model"] == best["model"],
                    }
                )
        save_json(
            {
                "selection_rule": "highest validation macro F1; test metrics are not used for selection",
                "selected_model": best["model"],
                "selected_checkpoint": best["checkpoint"],
                "results": results,
            },
            config.output.artifact_dir / "classifier_selection.json",
        )
    return results
