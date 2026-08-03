from __future__ import annotations

import cv2
import numpy as np

from .morphology import clean_binary_mask


class MOG2MotionDetector:
    """Foreground evidence for stationary-camera sequences."""

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        morphology_kernel: int = 3,
    ) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._kernel = morphology_kernel

    def apply(self, frame: np.ndarray, learning_rate: float = -1.0) -> np.ndarray:
        mask = self._subtractor.apply(frame, learningRate=learning_rate)
        if np.any(mask == 127):
            mask = np.where(mask == 255, 255, 0).astype(np.uint8)
        return clean_binary_mask(mask, self._kernel, 1, 1)
