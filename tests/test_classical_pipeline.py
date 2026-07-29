from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.classical.config import load_config
from firevision.classical.features import extract_handcrafted_features
from firevision.classical.pipeline import run_pipeline


def _write_fire(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.full((128, 128, 3), 15, dtype=np.uint8)
    points = np.array([[42, 98], [50, 45], [64, 70], [76, 30], [88, 98]], dtype=np.int32)
    cv2.fillPoly(image, [points], (20, 115 + seed % 40, 245))
    cv2.circle(image, (64, 78), 20, (10, 190, 255), -1)
    noise = rng.integers(0, 8, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _write_smoke(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.full((128, 128, 3), 25, dtype=np.uint8)
    for centre, radius, intensity in [((52, 84), 25, 150), ((72, 63), 28, 175), ((61, 38), 22, 195)]:
        cv2.circle(image, centre, radius, (intensity, intensity, intensity), -1)
    image = cv2.GaussianBlur(image, (15, 15), 0)
    noise = rng.integers(0, 7, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _write_normal(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :] = (150 + seed % 40, 70, 20)
    cv2.rectangle(image, (20, 20), (108, 108), (190, 90, 25), 4)
    cv2.line(image, (0, 0), (127, 127), (220, 120, 40), 3)
    noise = rng.integers(0, 8, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _build_dataset(root: Path) -> None:
    counts = {"train": 10, "val": 5, "test": 5}
    for split, count in counts.items():
        image_dir = root / "processed" / "images" / split
        label_dir = root / "processed" / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            fire_path = image_dir / f"fire_{index:03d}.jpg"
            smoke_path = image_dir / f"smoke_{index:03d}.jpg"
            normal_path = image_dir / f"normal_{index:03d}.jpg"
            _write_fire(fire_path, index)
            _write_smoke(smoke_path, index + 100)
            _write_normal(normal_path, index + 200)
            (label_dir / f"fire_{index:03d}.txt").write_text(
                "0 0.5 0.55 0.5 0.65\n", encoding="utf-8"
            )
            (label_dir / f"smoke_{index:03d}.txt").write_text(
                "1 0.5 0.5 0.65 0.75\n", encoding="utf-8"
            )
            (label_dir / f"normal_{index:03d}.txt").write_text("", encoding="utf-8")


def test_classical_end_to_end(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    config_data = {
        "project_root": str(tmp_path),
        "seed": 42,
        "dataset_dir": "processed",
        "classes": {0: "fire", 1: "smoke"},
        "patches": {
            "size": 64,
            "context": 0.05,
            "jpeg_quality": 95,
            "min_box_pixels": 10,
            "normal_per_negative_image": 1,
            "normal_per_positive_image": 0,
            "normal_crop_min_fraction": 0.6,
            "normal_crop_max_fraction": 0.9,
            "normal_max_iou": 0.0,
            "max_per_class": {
                "train": {"fire": 10, "smoke": 10, "normal": 10},
                "val": {"fire": 5, "smoke": 5, "normal": 5},
                "test": {"fire": 5, "smoke": 5, "normal": 5},
            },
        },
        "colour_baseline": {
            "gaussian_kernel": 3,
            "morphology_kernel": 3,
            "opening_iterations": 1,
            "closing_iterations": 1,
            "pixel_samples_per_image": 256,
            "maximum_pixels_per_class": 10000,
            "fire_vote_threshold": 4,
            "smoke_vote_threshold": 5,
            "area_grid_fire": [0.005, 0.02],
            "area_grid_smoke": [0.01, 0.04],
        },
        "features": {
            "image_size": 64,
            "hog_orientations": 9,
            "hog_pixels_per_cell": [8, 8],
            "hog_cells_per_block": [2, 2],
            "lbp_points": 8,
            "lbp_radius": 1,
            "histogram_bins": 8,
            "include_contour_statistics": True,
        },
        "svm": {
            "c_values": [1.0],
            "gamma_values": ["scale"],
            "cache_size_mb": 256,
            "max_iter": -1,
            "probability": False,
        },
        "motion": {
            "mog2_history": 50,
            "mog2_var_threshold": 16.0,
            "mog2_detect_shadows": False,
            "farneback_pyr_scale": 0.5,
            "farneback_levels": 2,
            "farneback_winsize": 11,
            "farneback_iterations": 2,
            "farneback_poly_n": 5,
            "farneback_poly_sigma": 1.2,
            "moving_magnitude_threshold": 1.0,
        },
        "output": {
            "patch_dir": "patches",
            "report_dir": "reports",
            "artifact_dir": "artifacts",
            "overwrite": True,
        },
    }
    config_path = tmp_path / "configs" / "classical.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)
    summary = run_pipeline(config)
    assert summary["patch_counts"]["train"] == {"fire": 10, "smoke": 10, "normal": 10}
    assert (tmp_path / "artifacts" / "colour_thresholds.json").exists()
    assert (tmp_path / "artifacts" / "hog_lbp_colour_rbf_svm.joblib").exists()
    assert (tmp_path / "reports" / "REPORT.md").exists()
    assert (tmp_path / "reports" / "method_comparison.csv").exists()
    assert summary["svm_test"]["support"] == 15


def test_feature_extraction_is_deterministic() -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(image, (48, 48), 20, (20, 120, 245), -1)
    from firevision.classical.config import FeatureConfig

    config = FeatureConfig(
        image_size=64,
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        hog_cells_per_block=(2, 2),
        lbp_points=8,
        lbp_radius=1,
        histogram_bins=8,
        include_contour_statistics=True,
    )
    first = extract_handcrafted_features(image, config)
    second = extract_handcrafted_features(image, config)
    assert first.shape == second.shape
    assert np.array_equal(first, second)
