from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ModelConfig:
    enabled: bool
    pretrained: bool
    batch_size: int
    stage1_epochs: int
    stage2_epochs: int
    unfreeze_last_blocks: int
    head_learning_rate: float
    finetune_learning_rate: float


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    image_size: int
    num_workers: int
    weight_decay: float
    label_smoothing: float
    patience: int
    amp: bool
    models: dict[str, ModelConfig]


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    base_model: str
    image_sizes: tuple[int, ...]
    epochs: int
    batch_size: int
    workers: int
    patience: int
    amp: bool
    cache: bool | str
    optimizer: str
    learning_rate: float | None
    weight_decay: float
    close_mosaic: int
    confidence_grid: tuple[float, ...]
    iou_grid: tuple[float, ...]
    max_det: int


@dataclass(frozen=True, slots=True)
class OutputConfig:
    artifact_dir: Path
    report_dir: Path
    yolo_project_dir: Path
    overwrite: bool


@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    project_root: Path
    seed: int
    device: str
    detection_dataset_yaml: Path
    classification_patch_dir: Path
    classifier_classes: tuple[str, ...]
    detector_classes: dict[int, str]
    classifier: ClassifierConfig
    detector: DetectorConfig
    output: OutputConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_model_config(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        enabled=bool(raw.get("enabled", True)),
        pretrained=bool(raw.get("pretrained", True)),
        batch_size=int(raw["batch_size"]),
        stage1_epochs=int(raw["stage1_epochs"]),
        stage2_epochs=int(raw["stage2_epochs"]),
        unfreeze_last_blocks=int(raw["unfreeze_last_blocks"]),
        head_learning_rate=float(raw["head_learning_rate"]),
        finetune_learning_rate=float(raw["finetune_learning_rate"]),
    )


def load_config(config_path: str | Path) -> DetectorTrainingConfig:
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Detector Training configuration must be a YAML mapping")

    configured_root = Path(raw.get("project_root", ".")).expanduser()
    root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (config_path.parent.parent / configured_root).resolve()
    )

    classifier_raw = raw["classifier"]
    detector_raw = raw["detector"]
    output_raw = raw["output"]

    model_configs = {
        str(name): _load_model_config(values)
        for name, values in classifier_raw["models"].items()
    }
    supported = {"mobilenet_v3_small", "vgg16"}
    unknown = set(model_configs) - supported
    if unknown:
        raise ValueError(f"Unsupported classifier models: {sorted(unknown)}")

    image_sizes = tuple(int(value) for value in detector_raw["image_sizes"])
    if not image_sizes:
        raise ValueError("detector.image_sizes cannot be empty")
    if any(value < 320 or value % 32 != 0 for value in image_sizes):
        raise ValueError("Every detector image size must be >=320 and divisible by 32")

    classifier_classes = tuple(str(value) for value in raw["classifier_classes"])
    if classifier_classes != ("fire", "smoke", "normal"):
        raise ValueError(
            "Detector Training classifier class order must be ['fire', 'smoke', 'normal']"
        )
    detector_classes = {int(key): str(value) for key, value in raw["detector_classes"].items()}
    if detector_classes != {0: "fire", 1: "smoke"}:
        raise ValueError("Detector Training detector classes must be {0: fire, 1: smoke}")

    cache_value = detector_raw.get("cache", False)
    if not isinstance(cache_value, (bool, str)):
        raise ValueError("detector.cache must be a boolean or Ultralytics cache mode")

    config = DetectorTrainingConfig(
        project_root=root,
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        detection_dataset_yaml=_resolve(root, raw["detection_dataset_yaml"]),
        classification_patch_dir=_resolve(root, raw["classification_patch_dir"]),
        classifier_classes=classifier_classes,
        detector_classes=detector_classes,
        classifier=ClassifierConfig(
            image_size=int(classifier_raw["image_size"]),
            num_workers=int(classifier_raw.get("num_workers", 2)),
            weight_decay=float(classifier_raw.get("weight_decay", 1e-4)),
            label_smoothing=float(classifier_raw.get("label_smoothing", 0.0)),
            patience=int(classifier_raw.get("patience", 5)),
            amp=bool(classifier_raw.get("amp", True)),
            models=model_configs,
        ),
        detector=DetectorConfig(
            base_model=str(detector_raw["base_model"]),
            image_sizes=image_sizes,
            epochs=int(detector_raw["epochs"]),
            batch_size=int(detector_raw["batch_size"]),
            workers=int(detector_raw.get("workers", 2)),
            patience=int(detector_raw.get("patience", 12)),
            amp=bool(detector_raw.get("amp", True)),
            cache=cache_value,
            optimizer=str(detector_raw.get("optimizer", "AdamW")),
            learning_rate=(
                None
                if detector_raw.get("learning_rate") in (None, "auto")
                else float(detector_raw["learning_rate"])
            ),
            weight_decay=float(detector_raw.get("weight_decay", 5e-4)),
            close_mosaic=int(detector_raw.get("close_mosaic", 10)),
            confidence_grid=tuple(float(v) for v in detector_raw["confidence_grid"]),
            iou_grid=tuple(float(v) for v in detector_raw["iou_grid"]),
            max_det=int(detector_raw.get("max_det", 100)),
        ),
        output=OutputConfig(
            artifact_dir=_resolve(root, output_raw["artifact_dir"]),
            report_dir=_resolve(root, output_raw["report_dir"]),
            yolo_project_dir=_resolve(root, output_raw["yolo_project_dir"]),
            overwrite=bool(output_raw.get("overwrite", True)),
        ),
    )

    if config.classifier.image_size < 96:
        raise ValueError("classifier.image_size must be at least 96")
    for name, model in config.classifier.models.items():
        if model.batch_size < 1:
            raise ValueError(f"{name}.batch_size must be positive")
        if model.stage1_epochs < 0 or model.stage2_epochs < 0:
            raise ValueError(f"{name} epoch counts cannot be negative")
        if model.stage1_epochs + model.stage2_epochs < 1 and model.enabled:
            raise ValueError(f"Enabled model {name} must train for at least one epoch")
    if not config.detector.confidence_grid or not config.detector.iou_grid:
        raise ValueError("Detector confidence and IoU grids cannot be empty")
    return config
