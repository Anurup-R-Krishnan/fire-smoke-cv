from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.detector.classifier_data import PatchFolderDataset, build_transforms
from firevision.detector.classifier_models import build_classifier, freeze_backbone, parameter_counts
from firevision.detector.config import load_config
from firevision.detector.detector import (
    normalise_detector_metrics,
    select_best_threshold,
    select_best_trial,
)
from firevision.detector.gradcam import GradCAM, overlay_heatmap


def _write_patch(path: Path, class_name: str) -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    if class_name == "fire":
        cv2.circle(image, (48, 58), 24, (10, 120, 250), -1)
    elif class_name == "smoke":
        cv2.circle(image, (48, 50), 28, (170, 170, 170), -1)
        image = cv2.GaussianBlur(image, (11, 11), 0)
    else:
        image[:] = (130, 70, 25)
        cv2.rectangle(image, (15, 15), (80, 80), (190, 100, 35), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _detector_config(tmp_path: Path) -> Path:
    for split in ("train", "val", "test"):
        for class_name in ("fire", "smoke", "normal"):
            _write_patch(tmp_path / "patches" / split / class_name / "one.jpg", class_name)
    detection_root = tmp_path / "detector"
    detection_root.mkdir()
    (detection_root / "data.yaml").write_text("names: [fire, smoke]\n", encoding="utf-8")
    raw = {
        "project_root": str(tmp_path),
        "seed": 42,
        "device": "cpu",
        "detection_dataset_yaml": "detector/data.yaml",
        "classification_patch_dir": "patches",
        "classifier_classes": ["fire", "smoke", "normal"],
        "detector_classes": {0: "fire", 1: "smoke"},
        "classifier": {
            "image_size": 96,
            "num_workers": 0,
            "weight_decay": 0.0001,
            "label_smoothing": 0.0,
            "patience": 1,
            "amp": False,
            "models": {
                "mobilenet_v3_small": {
                    "enabled": True,
                    "pretrained": False,
                    "batch_size": 2,
                    "stage1_epochs": 1,
                    "stage2_epochs": 0,
                    "unfreeze_last_blocks": 0,
                    "head_learning_rate": 0.001,
                    "finetune_learning_rate": 0.0001,
                },
                "vgg16": {
                    "enabled": False,
                    "pretrained": False,
                    "batch_size": 1,
                    "stage1_epochs": 1,
                    "stage2_epochs": 0,
                    "unfreeze_last_blocks": 0,
                    "head_learning_rate": 0.001,
                    "finetune_learning_rate": 0.0001,
                },
            },
        },
        "detector": {
            "base_model": "yolo11n.pt",
            "image_sizes": [512, 640],
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "patience": 1,
            "amp": False,
            "cache": False,
            "optimizer": "AdamW",
            "learning_rate": "auto",
            "weight_decay": 0.0005,
            "close_mosaic": 1,
            "confidence_grid": [0.2, 0.3],
            "iou_grid": [0.4, 0.5],
            "max_det": 50,
        },
        "output": {
            "artifact_dir": "artifacts",
            "report_dir": "reports",
            "yolo_project_dir": "runs",
            "overwrite": True,
        },
    }
    path = tmp_path / "configs" / "detector.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_detector_config_dataset_and_model(tmp_path: Path) -> None:
    config = load_config(_detector_config(tmp_path))
    train_transform, evaluation_transform = build_transforms(config.classifier.image_size)
    dataset = PatchFolderDataset(
        config.classification_patch_dir,
        "train",
        config.classifier_classes,
        evaluation_transform,
    )
    assert [sample.class_name for sample in dataset.samples] == ["fire", "smoke", "normal"]
    tensor, class_id, path = dataset[0]
    assert tensor.shape == (3, 96, 96)
    assert class_id == 0
    assert Path(path).exists()

    bundle = build_classifier("mobilenet_v3_small", 3, pretrained=False)
    freeze_backbone("mobilenet_v3_small", bundle.model)
    counts = parameter_counts(bundle.model)
    assert counts["trainable"] < counts["total"]
    bundle.model.eval()
    with torch.no_grad():
        output = bundle.model(tensor.unsqueeze(0))
    assert output.shape == (1, 3)


def test_gradcam_produces_normalised_overlay() -> None:
    class TinyCAMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            features = self.features(tensor)
            return self.classifier(self.pool(features).flatten(1))

    model = TinyCAMModel().eval()
    cam = GradCAM(model, model.features[0])
    try:
        tensor = torch.rand(1, 3, 32, 32, requires_grad=True)
        heatmap, predicted = cam.generate(tensor)
    finally:
        cam.close()
    assert predicted in {0, 1, 2}
    assert heatmap.ndim == 2
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0 + 1e-6
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    overlay = overlay_heatmap(image, heatmap)
    assert overlay.shape == image.shape


class _Metrics:
    results_dict = {
        "metrics/precision(B)": 0.75,
        "metrics/recall(B)": 0.60,
        "metrics/mAP50(B)": 0.70,
        "metrics/mAP50-95(B)": 0.42,
    }


def test_detector_selection_is_validation_driven() -> None:
    metrics = normalise_detector_metrics(_Metrics())
    assert abs(float(metrics["f1"]) - (2 * 0.75 * 0.60 / 1.35)) < 1e-8
    trials = [
        {
            "status": "completed",
            "image_size": 512,
            "validation_metrics": {"map50_95": 0.40, "map50": 0.70, "f1": 0.66},
        },
        {
            "status": "completed",
            "image_size": 640,
            "validation_metrics": {"map50_95": 0.45, "map50": 0.68, "f1": 0.64},
        },
    ]
    assert select_best_trial(trials)["image_size"] == 640
    thresholds = [
        {"confidence": 0.2, "iou": 0.5, "precision": 0.6, "recall": 0.8, "f1": 0.685},
        {"confidence": 0.3, "iou": 0.5, "precision": 0.75, "recall": 0.7, "f1": 0.724},
    ]
    assert select_best_threshold(thresholds)["confidence"] == 0.3


def test_classifier_training_loop_with_tiny_model(tmp_path: Path, monkeypatch) -> None:
    from firevision.detector import classifier_train as training_module
    from firevision.detector.classifier_models import ClassifierBundle

    config = load_config(_detector_config(tmp_path))

    class TinyClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3))

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(tensor))

    def fake_builder(_name: str, class_count: int, pretrained: bool) -> ClassifierBundle:
        assert class_count == 3
        assert pretrained is False
        model = TinyClassifier()
        return ClassifierBundle(model=model, target_layer=model.features[0])

    monkeypatch.setattr(training_module, "build_classifier", fake_builder)
    result = training_module.train_classifier(config, "mobilenet_v3_small")
    assert result["status"] == "completed"
    assert result["test_metrics"]["support"] == 3
    assert (tmp_path / "artifacts" / "mobilenet_v3_small_best.pt").exists()
    assert (tmp_path / "reports" / "mobilenet_v3_small" / "test_metrics.json").exists()
    assert (tmp_path / "reports" / "mobilenet_v3_small" / "training_curves.png").exists()
