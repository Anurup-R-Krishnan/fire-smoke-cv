from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.data.config import load_config
from firevision.data.pipeline import run_pipeline


def _write_image(path: Path, index: int) -> None:
    image = np.full((120, 160, 3), 30 + index * 3, dtype=np.uint8)
    cv2.circle(image, (40 + index, 60), 20, (20, 80 + index, 220), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_data_end_to_end(tmp_path: Path) -> None:
    images = tmp_path / "raw" / "images"
    labels = tmp_path / "raw" / "labels"
    labels.mkdir(parents=True)

    for index in range(24):
        image_path = images / f"sample_{index:03d}.jpg"
        _write_image(image_path, index)
        label_path = labels / f"sample_{index:03d}.txt"
        if index % 4 == 0:
            label_path.write_text("", encoding="utf-8")
        elif index % 4 == 1:
            label_path.write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")
        elif index % 4 == 2:
            label_path.write_text("1 0.5 0.5 0.4 0.3\n", encoding="utf-8")
        else:
            label_path.write_text(
                "0 0.45 0.5 0.2 0.3\n1 0.65 0.5 0.2 0.3\n", encoding="utf-8"
            )

    # Exact duplicate with a different filename should be removed.
    duplicate = images / "duplicate.jpg"
    duplicate.write_bytes((images / "sample_001.jpg").read_bytes())
    (labels / "duplicate.txt").write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")

    config_data = {
        "project_root": str(tmp_path),
        "seed": 42,
        "classes": {0: "fire", 1: "smoke"},
        "sources": [
            {
                "name": "synthetic",
                "format": "yolo",
                "images_dir": "raw/images",
                "labels_dir": "raw/labels",
                "class_map": {0: 0, 1: 1},
                "missing_label_is_negative": True,
            }
        ],
        "validation": {
            "min_width": 32,
            "min_height": 32,
            "repair_out_of_bounds_boxes": True,
            "drop_unknown_classes": True,
            "drop_tiny_boxes_pixels": 4,
        },
        "deduplication": {
            "exact_duplicates": True,
            "perceptual_hash": True,
            "phash_hamming_threshold": 2,
        },
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "stratify_by": "class_signature"},
        "output": {
            "dataset_dir": "processed",
            "report_dir": "reports",
            "transfer_mode": "copy",
            "overwrite": True,
            "preview_samples": 8,
        },
    }
    config_path = tmp_path / "configs" / "data.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    summary = run_pipeline(load_config(config_path))
    assert summary["total_retained_images"] == 24
    assert sum(summary["split_counts"].values()) == 24
    assert (tmp_path / "processed" / "data.yaml").exists()
    assert (tmp_path / "reports" / "preview.jpg").exists()

    with (tmp_path / "reports" / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 24

    groups: dict[str, set[str]] = {}
    for row in manifest:
        groups.setdefault(row["duplicate_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
