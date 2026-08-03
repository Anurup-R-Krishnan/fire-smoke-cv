from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import MotionConfig


@dataclass(frozen=True, slots=True)
class FlowStatistics:
    mean_magnitude: float
    magnitude_variance: float
    moving_pixel_ratio: float
    dominant_angle_degrees: float
    upward_motion_ratio: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def farneback_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    config: MotionConfig,
) -> np.ndarray:
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Optical-flow frames must have the same dimensions")
    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        config.farneback_pyr_scale,
        config.farneback_levels,
        config.farneback_winsize,
        config.farneback_iterations,
        config.farneback_poly_n,
        config.farneback_poly_sigma,
        0,
    )


def flow_statistics(
    flow: np.ndarray,
    magnitude_threshold: float,
    region_mask: np.ndarray | None = None,
) -> FlowStatistics:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Flow must have shape H x W x 2")
    horizontal = flow[..., 0]
    vertical = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(horizontal, vertical, angleInDegrees=True)
    valid = np.ones(magnitude.shape, dtype=bool)
    if region_mask is not None:
        if region_mask.shape != magnitude.shape:
            raise ValueError("region_mask shape must match optical flow")
        valid &= region_mask > 0
    values = magnitude[valid]
    if values.size == 0:
        return FlowStatistics(0.0, 0.0, 0.0, 0.0, 0.0)
    moving = valid & (magnitude >= magnitude_threshold)
    moving_count = int(np.count_nonzero(moving))
    valid_count = int(np.count_nonzero(valid))
    if moving_count:
        radians = np.deg2rad(angle[moving])
        dominant = np.rad2deg(
            np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
        ) % 360.0
        upward_ratio = float(np.mean(vertical[moving] < 0.0))
    else:
        dominant = 0.0
        upward_ratio = 0.0
    return FlowStatistics(
        mean_magnitude=float(np.mean(values)),
        magnitude_variance=float(np.var(values)),
        moving_pixel_ratio=moving_count / max(1, valid_count),
        dominant_angle_degrees=float(dominant),
        upward_motion_ratio=upward_ratio,
    )
