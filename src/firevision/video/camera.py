from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameUndistorter:
    def __init__(self, enabled: bool, calibration_file: Path) -> None:
        self.enabled = enabled
        self.camera_matrix: np.ndarray | None = None
        self.distortion: np.ndarray | None = None
        self._maps: tuple[np.ndarray, np.ndarray] | None = None
        self._shape: tuple[int, int] | None = None
        if not enabled:
            return
        if not calibration_file.exists():
            raise FileNotFoundError(
                f"Camera undistortion is enabled but calibration is missing: {calibration_file}"
            )
        data = np.load(calibration_file)
        matrix_key = "camera_matrix" if "camera_matrix" in data else "mtx"
        distortion_key = (
            "distortion"
            if "distortion" in data
            else ("dist_coeffs" if "dist_coeffs" in data else "dist")
        )
        if matrix_key not in data or distortion_key not in data:
            raise ValueError(
                "Calibration NPZ must contain camera_matrix plus distortion/dist_coeffs, or mtx/dist arrays"
            )
        self.camera_matrix = np.asarray(data[matrix_key], dtype=np.float64)
        self.distortion = np.asarray(data[distortion_key], dtype=np.float64)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        height, width = frame.shape[:2]
        shape = (width, height)
        if self._maps is None or self._shape != shape:
            assert self.camera_matrix is not None and self.distortion is not None
            map_x, map_y = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion,
                None,
                self.camera_matrix,
                shape,
                cv2.CV_32FC1,
            )
            self._maps = (map_x, map_y)
            self._shape = shape
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
