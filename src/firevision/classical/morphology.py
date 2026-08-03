from __future__ import annotations

import cv2
import numpy as np


def clean_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
) -> np.ndarray:
    """Remove isolated pixels and close small holes in an 8-bit binary mask."""
    if mask.ndim != 2:
        raise ValueError("Binary mask must be a single-channel image")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if opening_iterations > 0:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=opening_iterations
        )
    if closing_iterations > 0:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations
        )
    return binary
