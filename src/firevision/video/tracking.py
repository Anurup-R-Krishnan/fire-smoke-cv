from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .models import Detection, TrackObservation


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_to_measurement(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float64)


def _measurement_to_bbox(measurement: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, width, height = measurement[:4]
    width, height = max(1.0, float(width)), max(1.0, float(height))
    return (float(cx - width / 2), float(cy - height / 2), float(cx + width / 2), float(cy + height / 2))


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over centre, width, and height."""

    def __init__(self, bbox: tuple[float, float, float, float], config: TrackerConfig) -> None:
        self.state = np.zeros(8, dtype=np.float64)
        self.state[:4] = _bbox_to_measurement(bbox)
        self.transition = np.eye(8, dtype=np.float64)
        self.transition[:4, 4:] = np.eye(4, dtype=np.float64)
        self.measurement_matrix = np.zeros((4, 8), dtype=np.float64)
        self.measurement_matrix[:, :4] = np.eye(4, dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64) * config.initial_covariance
        self.process_noise = np.eye(8, dtype=np.float64) * config.process_noise
        self.measurement_noise = np.eye(4, dtype=np.float64) * config.measurement_noise

    def predict(self) -> tuple[float, float, float, float]:
        self.state = self.transition @ self.state
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return _measurement_to_bbox(self.state)

    def update(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        measurement = _bbox_to_measurement(bbox)
        residual = measurement - self.measurement_matrix @ self.state
        innovation = self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + self.measurement_noise
        gain = self.covariance @ self.measurement_matrix.T @ np.linalg.pinv(innovation)
        self.state = self.state + gain @ residual
        identity = np.eye(8, dtype=np.float64)
        self.covariance = (identity - gain @ self.measurement_matrix) @ self.covariance
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)
        return _measurement_to_bbox(self.state)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _measurement_to_bbox(self.state)


@dataclass(slots=True)
class _Track:
    track_id: int
    class_id: int
    label: str
    filter: KalmanBoxFilter
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    observed: bool = True
    center_history: list[tuple[int, int]] = field(default_factory=list)

    def append_center(self, limit: int = 50) -> None:
        x1, y1, x2, y2 = self.filter.bbox
        self.center_history.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        if len(self.center_history) > limit:
            del self.center_history[:-limit]


class SortTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackObservation]:
        for track in self._tracks:
            track.filter.predict()
            track.age += 1
            track.time_since_update += 1
            track.observed = False

        matches, unmatched_tracks, unmatched_detections = self._associate(detections)
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            track.filter.update(detection.bbox)
            track.confidence = detection.confidence
            track.hits += 1
            track.time_since_update = 0
            track.observed = True
            track.append_center()

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = _Track(
                track_id=self._next_id,
                class_id=detection.class_id,
                label=detection.label,
                filter=KalmanBoxFilter(detection.bbox, self.config),
                confidence=detection.confidence,
            )
            track.append_center()
            self._next_id += 1
            self._tracks.append(track)

        self._tracks = [track for track in self._tracks if track.time_since_update <= self.config.max_age]
        observations: list[TrackObservation] = []
        for track in self._tracks:
            if track.hits < self.config.minimum_hits and track.time_since_update > 0:
                continue
            observations.append(
                TrackObservation(
                    track_id=track.track_id,
                    bbox=track.filter.bbox,
                    class_id=track.class_id,
                    label=track.label,
                    confidence=track.confidence,
                    observed=track.observed,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update,
                    center_history=list(track.center_history),
                )
            )
        return observations

    def _associate(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))
        cost = np.full((len(self._tracks), len(detections)), 1e6, dtype=np.float64)
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                cost[track_index, detection_index] = 1.0 - bbox_iou(track.filter.bbox, detection.bbox)
        row_indices, column_indices = linear_sum_assignment(cost)
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for track_index, detection_index in zip(row_indices.tolist(), column_indices.tolist()):
            if cost[track_index, detection_index] >= 1e5:
                continue
            overlap = 1.0 - cost[track_index, detection_index]
            if overlap < self.config.minimum_iou:
                continue
            matches.append((track_index, detection_index))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
        unmatched_tracks = [index for index in range(len(self._tracks)) if index not in matched_tracks]
        unmatched_detections = [index for index in range(len(detections)) if index not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections
