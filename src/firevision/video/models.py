from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True, slots=True)
class MotionEvidence:
    mean_magnitude: float = 0.0
    moving_pixel_ratio: float = 0.0
    upward_motion_ratio: float = 0.0
    foreground_ratio: float = 0.0
    dominant_dx: float = 0.0
    dominant_dy: float = 0.0
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class TemporalEvidence:
    state: str
    observed_hits: int
    window_length: int
    persistence: float
    smoothed_confidence: float
    confirmed: bool


@dataclass(frozen=True, slots=True)
class RiskEvidence:
    score: float
    level: str
    confidence_component: float
    persistence_component: float
    motion_component: float
    area_component: float
    growth_component: float


@dataclass(slots=True)
class TrackObservation:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float
    observed: bool
    age: int
    hits: int
    time_since_update: int
    center_history: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FusedTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    label: str
    observed: bool
    temporal: TemporalEvidence
    motion: MotionEvidence
    risk: RiskEvidence
    area_ratio: float
    growth_ratio: float

    def as_row(self, frame_index: int, timestamp_seconds: float) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "frame_index": frame_index,
            "timestamp_seconds": round(timestamp_seconds, 6),
            "track_id": self.track_id,
            "class_id": self.class_id,
            "label": self.label,
            "observed": int(self.observed),
            "x1": round(x1, 3),
            "y1": round(y1, 3),
            "x2": round(x2, 3),
            "y2": round(y2, 3),
            "state": self.temporal.state,
            "confirmed": int(self.temporal.confirmed),
            "smoothed_confidence": round(self.temporal.smoothed_confidence, 6),
            "persistence": round(self.temporal.persistence, 6),
            "motion_score": round(self.motion.score, 6),
            "mean_motion": round(self.motion.mean_magnitude, 6),
            "moving_pixel_ratio": round(self.motion.moving_pixel_ratio, 6),
            "upward_motion_ratio": round(self.motion.upward_motion_ratio, 6),
            "foreground_ratio": round(self.motion.foreground_ratio, 6),
            "area_ratio": round(self.area_ratio, 6),
            "growth_ratio": round(self.growth_ratio, 6),
            "risk_score": round(self.risk.score, 6),
            "risk_level": self.risk.level,
        }
