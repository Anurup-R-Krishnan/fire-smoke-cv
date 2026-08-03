from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class RegionStatistics:
    area_ratio: float
    largest_area_ratio: float
    largest_perimeter_ratio: float
    largest_aspect_ratio: float
    largest_solidity: float
    largest_extent: float
    component_count: int
    centroid_x: float
    centroid_y: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def mask_region_statistics(mask: np.ndarray, minimum_relative_area: float = 0.0005) -> RegionStatistics:
    if mask.ndim != 2:
        raise ValueError("Mask must be single-channel")
    height, width = mask.shape
    frame_area = float(max(1, height * width))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    retained = [
        contour
        for contour in contours
        if cv2.contourArea(contour) / frame_area >= minimum_relative_area
    ]
    foreground_area = float(cv2.countNonZero(binary)) / frame_area
    if not retained:
        return RegionStatistics(
            area_ratio=foreground_area,
            largest_area_ratio=0.0,
            largest_perimeter_ratio=0.0,
            largest_aspect_ratio=0.0,
            largest_solidity=0.0,
            largest_extent=0.0,
            component_count=0,
            centroid_x=0.5,
            centroid_y=0.5,
        )

    largest = max(retained, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    x, y, box_width, box_height = cv2.boundingRect(largest)
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    moments = cv2.moments(largest)
    if moments["m00"]:
        centroid_x = float(moments["m10"] / moments["m00"]) / max(1, width)
        centroid_y = float(moments["m01"] / moments["m00"]) / max(1, height)
    else:
        centroid_x = (x + box_width / 2.0) / max(1, width)
        centroid_y = (y + box_height / 2.0) / max(1, height)

    return RegionStatistics(
        area_ratio=foreground_area,
        largest_area_ratio=area / frame_area,
        largest_perimeter_ratio=perimeter / max(1.0, 2.0 * (width + height)),
        largest_aspect_ratio=box_width / max(1.0, float(box_height)),
        largest_solidity=area / hull_area if hull_area > 0 else 0.0,
        largest_extent=area / max(1.0, float(box_width * box_height)),
        component_count=len(retained),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
    )
