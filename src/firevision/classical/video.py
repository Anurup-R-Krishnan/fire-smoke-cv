from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .background import MOG2MotionDetector
from .colour_rules import ColourThresholdModel, generate_colour_masks
from .config import ClassicalMLConfig
from .contours import mask_region_statistics
from .optical_flow import farneback_flow, flow_statistics


@dataclass(frozen=True, slots=True)
class VideoEvidence:
    fire_area_ratio: float
    smoke_area_ratio: float
    foreground_ratio: float
    mean_flow: float
    moving_pixel_ratio: float
    upward_motion_ratio: float


class ClassicalVideoProcessor:
    def __init__(self, config: ClassicalMLConfig, model: ColourThresholdModel, fixed_camera: bool) -> None:
        self.config = config
        self.model = model
        self.fixed_camera = fixed_camera
        self.previous_gray: np.ndarray | None = None
        self.background = (
            MOG2MotionDetector(
                history=config.motion.mog2_history,
                var_threshold=config.motion.mog2_var_threshold,
                detect_shadows=config.motion.mog2_detect_shadows,
                morphology_kernel=config.colour.morphology_kernel,
            )
            if fixed_camera
            else None
        )

    @staticmethod
    def _draw_regions(frame: np.ndarray, mask: np.ndarray, label: str) -> None:
        height, width = mask.shape
        minimum_area = height * width * 0.001
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 255, 255), 2)
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, VideoEvidence, np.ndarray, np.ndarray]:
        fire_mask, smoke_mask = generate_colour_masks(frame, self.model, self.config.colour)
        candidate_mask = cv2.bitwise_or(fire_mask, smoke_mask)
        fire_stats = mask_region_statistics(fire_mask)
        smoke_stats = mask_region_statistics(smoke_mask)

        foreground_ratio = 0.0
        if self.background is not None:
            foreground = self.background.apply(frame)
            foreground_ratio = float(cv2.countNonZero(cv2.bitwise_and(foreground, candidate_mask))) / max(
                1, cv2.countNonZero(candidate_mask)
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_flow = 0.0
        moving_ratio = 0.0
        upward_ratio = 0.0
        if self.previous_gray is not None:
            flow = farneback_flow(self.previous_gray, gray, self.config.motion)
            stats = flow_statistics(
                flow,
                self.config.motion.moving_magnitude_threshold,
                candidate_mask,
            )
            mean_flow = stats.mean_magnitude
            moving_ratio = stats.moving_pixel_ratio
            upward_ratio = stats.upward_motion_ratio
        self.previous_gray = gray

        annotated = frame.copy()
        self._draw_regions(annotated, fire_mask, "FIRE COLOUR CANDIDATE")
        self._draw_regions(annotated, smoke_mask, "SMOKE COLOUR CANDIDATE")
        lines = [
            f"fire area: {fire_stats.area_ratio:.3f}",
            f"smoke area: {smoke_stats.area_ratio:.3f}",
            f"flow mean: {mean_flow:.2f}",
            f"moving ratio: {moving_ratio:.2f}",
        ]
        if self.fixed_camera:
            lines.append(f"MOG2 overlap: {foreground_ratio:.2f}")
        for index, text in enumerate(lines):
            cv2.putText(
                annotated,
                text,
                (12, 25 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        evidence = VideoEvidence(
            fire_area_ratio=fire_stats.area_ratio,
            smoke_area_ratio=smoke_stats.area_ratio,
            foreground_ratio=foreground_ratio,
            mean_flow=mean_flow,
            moving_pixel_ratio=moving_ratio,
            upward_motion_ratio=upward_ratio,
        )
        return annotated, evidence, fire_mask, smoke_mask
