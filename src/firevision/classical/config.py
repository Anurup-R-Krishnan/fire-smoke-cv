from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PatchConfig:
    size: int
    context: float
    jpeg_quality: int
    min_box_pixels: int
    normal_per_negative_image: int
    normal_per_positive_image: int
    normal_crop_min_fraction: float
    normal_crop_max_fraction: float
    normal_max_iou: float
    max_per_class: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class ColourConfig:
    gaussian_kernel: int
    morphology_kernel: int
    opening_iterations: int
    closing_iterations: int
    pixel_samples_per_image: int
    maximum_pixels_per_class: int
    fire_vote_threshold: int
    smoke_vote_threshold: int
    area_grid_fire: tuple[float, ...]
    area_grid_smoke: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    image_size: int
    hog_orientations: int
    hog_pixels_per_cell: tuple[int, int]
    hog_cells_per_block: tuple[int, int]
    lbp_points: int
    lbp_radius: int
    histogram_bins: int
    include_contour_statistics: bool


@dataclass(frozen=True, slots=True)
class SVMConfig:
    c_values: tuple[float, ...]
    gamma_values: tuple[str | float, ...]
    cache_size_mb: int
    max_iter: int
    probability: bool


@dataclass(frozen=True, slots=True)
class MotionConfig:
    mog2_history: int
    mog2_var_threshold: float
    mog2_detect_shadows: bool
    farneback_pyr_scale: float
    farneback_levels: int
    farneback_winsize: int
    farneback_iterations: int
    farneback_poly_n: int
    farneback_poly_sigma: float
    moving_magnitude_threshold: float


@dataclass(frozen=True, slots=True)
class OutputConfig:
    patch_dir: Path
    report_dir: Path
    artifact_dir: Path
    overwrite: bool


@dataclass(frozen=True, slots=True)
class ClassicalMLConfig:
    project_root: Path
    seed: int
    dataset_dir: Path
    classes: dict[int, str]
    patches: PatchConfig
    colour: ColourConfig
    features: FeatureConfig
    svm: SVMConfig
    motion: MotionConfig
    output: OutputConfig


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _gamma_values(values: list[Any]) -> tuple[str | float, ...]:
    parsed: list[str | float] = []
    for value in values:
        if isinstance(value, str):
            if value not in {"scale", "auto"}:
                try:
                    parsed.append(float(value))
                except ValueError as exc:
                    raise ValueError(f"Unsupported SVM gamma value: {value}") from exc
            else:
                parsed.append(value)
        else:
            parsed.append(float(value))
    return tuple(parsed)


def load_config(path: str | Path) -> ClassicalMLConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root_value = raw.get("project_root", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (config_path.parent.parent / root).resolve()

    patches = raw["patches"]
    colour = raw["colour_baseline"]
    features = raw["features"]
    svm = raw["svm"]
    motion = raw["motion"]
    output = raw["output"]

    kernel = int(colour["gaussian_kernel"])
    morphology_kernel = int(colour["morphology_kernel"])
    if kernel % 2 == 0 or kernel < 1:
        raise ValueError("colour_baseline.gaussian_kernel must be a positive odd integer")
    if morphology_kernel % 2 == 0 or morphology_kernel < 1:
        raise ValueError("colour_baseline.morphology_kernel must be a positive odd integer")

    max_per_class = {
        split: {name: int(count) for name, count in values.items()}
        for split, values in patches["max_per_class"].items()
    }

    config = ClassicalMLConfig(
        project_root=root,
        seed=int(raw.get("seed", 42)),
        dataset_dir=_resolve(root, raw["dataset_dir"]),
        classes={int(key): str(value) for key, value in raw["classes"].items()},
        patches=PatchConfig(
            size=int(patches["size"]),
            context=float(patches["context"]),
            jpeg_quality=int(patches.get("jpeg_quality", 95)),
            min_box_pixels=int(patches.get("min_box_pixels", 12)),
            normal_per_negative_image=int(patches["normal_per_negative_image"]),
            normal_per_positive_image=int(patches["normal_per_positive_image"]),
            normal_crop_min_fraction=float(patches["normal_crop_min_fraction"]),
            normal_crop_max_fraction=float(patches["normal_crop_max_fraction"]),
            normal_max_iou=float(patches["normal_max_iou"]),
            max_per_class=max_per_class,
        ),
        colour=ColourConfig(
            gaussian_kernel=kernel,
            morphology_kernel=morphology_kernel,
            opening_iterations=int(colour["opening_iterations"]),
            closing_iterations=int(colour["closing_iterations"]),
            pixel_samples_per_image=int(colour["pixel_samples_per_image"]),
            maximum_pixels_per_class=int(colour["maximum_pixels_per_class"]),
            fire_vote_threshold=int(colour["fire_vote_threshold"]),
            smoke_vote_threshold=int(colour["smoke_vote_threshold"]),
            area_grid_fire=tuple(float(v) for v in colour["area_grid_fire"]),
            area_grid_smoke=tuple(float(v) for v in colour["area_grid_smoke"]),
        ),
        features=FeatureConfig(
            image_size=int(features["image_size"]),
            hog_orientations=int(features["hog_orientations"]),
            hog_pixels_per_cell=tuple(int(v) for v in features["hog_pixels_per_cell"]),
            hog_cells_per_block=tuple(int(v) for v in features["hog_cells_per_block"]),
            lbp_points=int(features["lbp_points"]),
            lbp_radius=int(features["lbp_radius"]),
            histogram_bins=int(features["histogram_bins"]),
            include_contour_statistics=bool(features["include_contour_statistics"]),
        ),
        svm=SVMConfig(
            c_values=tuple(float(v) for v in svm["c_values"]),
            gamma_values=_gamma_values(svm["gamma_values"]),
            cache_size_mb=int(svm["cache_size_mb"]),
            max_iter=int(svm.get("max_iter", -1)),
            probability=bool(svm.get("probability", False)),
        ),
        motion=MotionConfig(
            mog2_history=int(motion["mog2_history"]),
            mog2_var_threshold=float(motion["mog2_var_threshold"]),
            mog2_detect_shadows=bool(motion["mog2_detect_shadows"]),
            farneback_pyr_scale=float(motion["farneback_pyr_scale"]),
            farneback_levels=int(motion["farneback_levels"]),
            farneback_winsize=int(motion["farneback_winsize"]),
            farneback_iterations=int(motion["farneback_iterations"]),
            farneback_poly_n=int(motion["farneback_poly_n"]),
            farneback_poly_sigma=float(motion["farneback_poly_sigma"]),
            moving_magnitude_threshold=float(motion["moving_magnitude_threshold"]),
        ),
        output=OutputConfig(
            patch_dir=_resolve(root, output["patch_dir"]),
            report_dir=_resolve(root, output["report_dir"]),
            artifact_dir=_resolve(root, output["artifact_dir"]),
            overwrite=bool(output.get("overwrite", True)),
        ),
    )

    expected_classes = {0: "fire", 1: "smoke"}
    if config.classes != expected_classes:
        raise ValueError(f"Classical ML expects classes {expected_classes}, got {config.classes}")
    if config.patches.size < 32:
        raise ValueError("Patch size must be at least 32")
    if not 0 <= config.patches.normal_max_iou < 1:
        raise ValueError("normal_max_iou must be in [0, 1)")
    return config
