from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from .config import FeatureConfig, ClassicalMLConfig
from .contours import mask_region_statistics
from .dataset import PatchRecord


def _normalised_histogram(channel: np.ndarray, bins: int, value_range: tuple[int, int]) -> np.ndarray:
    histogram = cv2.calcHist([channel], [0], None, [bins], list(value_range)).reshape(-1)
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram.astype(np.float32)


def _edge_statistics(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    statistics = mask_region_statistics(edges, minimum_relative_area=0.0001)
    return np.asarray(
        [
            statistics.area_ratio,
            statistics.largest_area_ratio,
            statistics.largest_perimeter_ratio,
            statistics.largest_aspect_ratio,
            statistics.largest_solidity,
            statistics.largest_extent,
            float(statistics.component_count),
            statistics.centroid_x,
            statistics.centroid_y,
        ],
        dtype=np.float32,
    )


def extract_handcrafted_features(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    resized = cv2.resize(
        image,
        (config.image_size, config.image_size),
        interpolation=cv2.INTER_AREA if max(image.shape[:2]) > config.image_size else cv2.INTER_LINEAR,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)

    hog_features = hog(
        gray,
        orientations=config.hog_orientations,
        pixels_per_cell=config.hog_pixels_per_cell,
        cells_per_block=config.hog_cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    lbp = local_binary_pattern(
        gray,
        P=config.lbp_points,
        R=config.lbp_radius,
        method="uniform",
    )
    lbp_bins = config.lbp_points + 2
    lbp_histogram, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    lbp_histogram = lbp_histogram.astype(np.float32)
    lbp_histogram /= max(1.0, float(lbp_histogram.sum()))

    colour_features: list[np.ndarray] = []
    for channel_index, channel in enumerate(cv2.split(hsv)):
        value_range = (0, 180) if channel_index == 0 else (0, 256)
        colour_features.append(_normalised_histogram(channel, config.histogram_bins, value_range))
    for channel in cv2.split(ycrcb):
        colour_features.append(_normalised_histogram(channel, config.histogram_bins, (0, 256)))

    channel_statistics = []
    for image_space in (hsv, ycrcb):
        for channel in cv2.split(image_space):
            channel_statistics.extend([float(channel.mean()) / 255.0, float(channel.std()) / 255.0])

    pieces = [
        hog_features,
        lbp_histogram,
        *colour_features,
        np.asarray(channel_statistics, dtype=np.float32),
    ]
    if config.include_contour_statistics:
        pieces.append(_edge_statistics(gray))
    return np.concatenate(pieces).astype(np.float32, copy=False)


def feature_dimension(config: FeatureConfig) -> int:
    dummy = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
    return int(extract_handcrafted_features(dummy, config).shape[0])


def extract_split_features(
    records: list[PatchRecord],
    split: str,
    config: ClassicalMLConfig,
) -> Path:
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"No patch records found for split: {split}")
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for record in selected:
        image = cv2.imread(str(record.patch_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read patch: {record.patch_path}")
        vectors.append(extract_handcrafted_features(image, config.features))
        labels.append(record.class_id)
        paths.append(str(record.patch_path))

    output_path = config.output.artifact_dir / f"features_{split}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.stack(vectors).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        paths=np.asarray(paths, dtype=str),
    )
    metadata_path = config.output.artifact_dir / "feature_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dimension": int(vectors[0].shape[0]),
                "image_size": config.features.image_size,
                "hog_orientations": config.features.hog_orientations,
                "hog_pixels_per_cell": list(config.features.hog_pixels_per_cell),
                "hog_cells_per_block": list(config.features.hog_cells_per_block),
                "lbp_points": config.features.lbp_points,
                "lbp_radius": config.features.lbp_radius,
                "histogram_bins": config.features.histogram_bins,
                "include_contour_statistics": config.features.include_contour_statistics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def load_feature_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    return archive["X"], archive["y"], archive["paths"]
