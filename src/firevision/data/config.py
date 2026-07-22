from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SourceConfig:
    name: str
    format: str
    images_dir: Path
    labels_dir: Path
    class_map: dict[Any, int]
    missing_label_is_negative: bool


@dataclass(frozen=True, slots=True)
class DataPrepConfig:
    project_root: Path
    seed: int
    classes: dict[int, str]
    sources: list[SourceConfig]
    validation: dict[str, Any]
    deduplication: dict[str, Any]
    split: dict[str, float | str]
    output: dict[str, Any]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(config_path: str | Path) -> DataPrepConfig:
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    configured_root = Path(raw.get("project_root", ".")).expanduser()
    project_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (config_path.parent.parent / configured_root).resolve()
    )

    classes = {int(k): str(v) for k, v in raw["classes"].items()}
    expected_ids = list(range(len(classes)))
    if sorted(classes) != expected_ids:
        raise ValueError(f"Class IDs must be contiguous from 0. Found: {sorted(classes)}")

    sources: list[SourceConfig] = []
    for source in raw.get("sources", []):
        fmt = str(source["format"]).lower()
        if fmt not in {"yolo", "voc"}:
            raise ValueError(f"Unsupported source format: {fmt}")
        class_map_raw = source.get("class_map", {})
        class_map: dict[Any, int] = {}
        for key, value in class_map_raw.items():
            parsed_key: Any = int(key) if fmt == "yolo" else str(key).strip().lower()
            class_map[parsed_key] = int(value)
        sources.append(
            SourceConfig(
                name=str(source["name"]),
                format=fmt,
                images_dir=_resolve(project_root, source["images_dir"]),
                labels_dir=_resolve(project_root, source["labels_dir"]),
                class_map=class_map,
                missing_label_is_negative=bool(
                    source.get("missing_label_is_negative", False)
                ),
            )
        )

    split = raw["split"]
    total = float(split["train"]) + float(split["val"]) + float(split["test"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, found {total}")

    output = dict(raw["output"])
    output["dataset_dir"] = _resolve(project_root, output["dataset_dir"])
    output["report_dir"] = _resolve(project_root, output["report_dir"])

    return DataPrepConfig(
        project_root=project_root,
        seed=int(raw.get("seed", 42)),
        classes=classes,
        sources=sources,
        validation=dict(raw.get("validation", {})),
        deduplication=dict(raw.get("deduplication", {})),
        split=dict(split),
        output=output,
    )
