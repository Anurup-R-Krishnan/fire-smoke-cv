from __future__ import annotations

import cv2
import numpy as np

from .config import MotionConfig
from .models import MotionEvidence


class MotionAnalyzer:
    def __init__(self, config: MotionConfig, fixed_camera: bool = False) -> None:
        self.config = config
        self.previous_gray: np.ndarray | None = None
        self.flow: np.ndarray | None = None
        self.foreground_mask: np.ndarray | None = None
        self._mog2 = None
        if fixed_camera and config.mog2.enabled_for_fixed_camera:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=config.mog2.history,
                varThreshold=config.mog2.variance_threshold,
                detectShadows=False,
            )

    def update_frame(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not self.config.enabled or self.previous_gray is None:
            self.flow = None
        else:
            cfg = self.config.farneback
            self.flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray,
                gray,
                None,
                cfg.pyr_scale,
                cfg.levels,
                cfg.winsize,
                cfg.iterations,
                cfg.poly_n,
                cfg.poly_sigma,
                0,
            )
        self.previous_gray = gray
        if self._mog2 is not None:
            mask = self._mog2.apply(frame, learningRate=self.config.mog2.learning_rate)
            self.foreground_mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
            )
        else:
            self.foreground_mask = None

    def evidence(self, bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> MotionEvidence:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
        if x2 <= x1 or y2 <= y1:
            return MotionEvidence()
        mean_magnitude = moving_ratio = upward_ratio = foreground_ratio = 0.0
        dx = dy = 0.0
        if self.flow is not None:
            roi = self.flow[y1:y2, x1:x2]
            if roi.size:
                horizontal = roi[..., 0]
                vertical = roi[..., 1]
                magnitude = np.sqrt(horizontal**2 + vertical**2)
                moving = magnitude >= self.config.magnitude_threshold
                mean_magnitude = float(np.mean(magnitude))
                moving_ratio = float(np.mean(moving))
                if np.any(moving):
                    upward_ratio = float(np.mean(vertical[moving] < 0.0))
                    dx = float(np.mean(horizontal[moving]))
                    dy = float(np.mean(vertical[moving]))
        if self.foreground_mask is not None:
            roi_mask = self.foreground_mask[y1:y2, x1:x2]
            if roi_mask.size:
                foreground_ratio = float(np.mean(roi_mask > 0))
        magnitude_component = min(1.0, mean_magnitude / max(self.config.magnitude_clip, 1e-6))
        score = 0.35 * magnitude_component + 0.35 * moving_ratio + 0.15 * upward_ratio + 0.15 * foreground_ratio
        return MotionEvidence(
            mean_magnitude=mean_magnitude,
            moving_pixel_ratio=moving_ratio,
            upward_motion_ratio=upward_ratio,
            foreground_ratio=foreground_ratio,
            dominant_dx=dx,
            dominant_dy=dy,
            score=float(np.clip(score, 0.0, 1.0)),
        )


def _clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )
