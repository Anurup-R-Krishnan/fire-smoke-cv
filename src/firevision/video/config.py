from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CameraConfig:
    undistort: bool
    calibration_file: Path


@dataclass(frozen=True, slots=True)
class ModelConfig:
    checkpoint: Path
    thresholds: Path
    fallback_image_size: int
    fallback_confidence: float
    fallback_iou: float
    max_det: int


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    minimum_iou: float
    max_age: int
    minimum_hits: int
    process_noise: float
    measurement_noise: float
    initial_covariance: float


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    window_size: int
    fire_minimum_hits: int
    smoke_minimum_hits: int
    confidence_alpha: float
    minimum_confirmed_confidence: float


@dataclass(frozen=True, slots=True)
class FarnebackConfig:
    pyr_scale: float
    levels: int
    winsize: int
    iterations: int
    poly_n: int
    poly_sigma: float


@dataclass(frozen=True, slots=True)
class MOG2Config:
    enabled_for_fixed_camera: bool
    history: int
    variance_threshold: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class MotionConfig:
    enabled: bool
    magnitude_threshold: float
    magnitude_clip: float
    farneback: FarnebackConfig
    mog2: MOG2Config


@dataclass(frozen=True, slots=True)
class RiskWeights:
    confidence: float
    persistence: float
    motion: float
    area: float
    growth: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    weights: RiskWeights
    area_saturation_ratio: float
    growth_saturation_ratio: float
    moderate_threshold: float
    high_threshold: float
    critical_threshold: float


@dataclass(frozen=True, slots=True)
class VideoConfig:
    output_codec: str
    fallback_fps: float
    display_scale: float
    draw_motion_vector: bool
    draw_track_trail: bool
    trail_length: int


@dataclass(frozen=True, slots=True)
class OutputConfig:
    report_dir: Path
    default_video: Path
    frame_log: Path
    event_log: Path
    summary: Path


@dataclass(frozen=True, slots=True)
class VideoFusionConfig:
    project_root: Path
    device: str
    camera: CameraConfig
    model: ModelConfig
    tracker: TrackerConfig
    temporal: TemporalConfig
    motion: MotionConfig
    risk: RiskConfig
    video: VideoConfig
    output: OutputConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _bounded(name: str, value: float, low: float, high: float) -> float:
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}; received {value}")
    return value


def load_config(config_path: str | Path) -> VideoFusionConfig:
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Video Fusion configuration must be a YAML mapping")

    configured_root = Path(raw.get("project_root", ".")).expanduser()
    root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (path.parent.parent / configured_root).resolve()
    )
    camera_raw = raw.get("camera", {})
    model_raw = raw["model"]
    tracker_raw = raw["tracker"]
    temporal_raw = raw["temporal"]
    motion_raw = raw["motion"]
    farneback_raw = motion_raw["farneback"]
    mog2_raw = motion_raw["mog2"]
    risk_raw = raw["risk"]
    weights_raw = risk_raw["weights"]
    video_raw = raw["video"]
    output_raw = raw["output"]

    weights = RiskWeights(
        confidence=float(weights_raw["confidence"]),
        persistence=float(weights_raw["persistence"]),
        motion=float(weights_raw["motion"]),
        area=float(weights_raw["area"]),
        growth=float(weights_raw["growth"]),
    )
    if abs(sum((weights.confidence, weights.persistence, weights.motion, weights.area, weights.growth)) - 1.0) > 1e-6:
        raise ValueError("Risk weights must sum to 1.0")

    config = VideoFusionConfig(
        project_root=root,
        device=str(raw.get("device", "auto")),
        camera=CameraConfig(
            undistort=bool(camera_raw.get("undistort", False)),
            calibration_file=_resolve(root, camera_raw.get("calibration_file", "artifacts/calibration/camera_calibration.npz")),
        ),
        model=ModelConfig(
            checkpoint=_resolve(root, model_raw["checkpoint"]),
            thresholds=_resolve(root, model_raw["thresholds"]),
            fallback_image_size=int(model_raw.get("fallback_image_size", 640)),
            fallback_confidence=_bounded("fallback_confidence", float(model_raw.get("fallback_confidence", 0.25)), 0.0, 1.0),
            fallback_iou=_bounded("fallback_iou", float(model_raw.get("fallback_iou", 0.45)), 0.0, 1.0),
            max_det=int(model_raw.get("max_det", 100)),
        ),
        tracker=TrackerConfig(
            minimum_iou=_bounded("minimum_iou", float(tracker_raw.get("minimum_iou", 0.2)), 0.0, 1.0),
            max_age=int(tracker_raw.get("max_age", 8)),
            minimum_hits=int(tracker_raw.get("minimum_hits", 2)),
            process_noise=float(tracker_raw.get("process_noise", 0.01)),
            measurement_noise=float(tracker_raw.get("measurement_noise", 0.1)),
            initial_covariance=float(tracker_raw.get("initial_covariance", 10.0)),
        ),
        temporal=TemporalConfig(
            window_size=int(temporal_raw.get("window_size", 8)),
            fire_minimum_hits=int(temporal_raw.get("fire_minimum_hits", 3)),
            smoke_minimum_hits=int(temporal_raw.get("smoke_minimum_hits", 4)),
            confidence_alpha=_bounded("confidence_alpha", float(temporal_raw.get("confidence_alpha", 0.4)), 0.0, 1.0),
            minimum_confirmed_confidence=_bounded(
                "minimum_confirmed_confidence",
                float(temporal_raw.get("minimum_confirmed_confidence", 0.25)),
                0.0,
                1.0,
            ),
        ),
        motion=MotionConfig(
            enabled=bool(motion_raw.get("enabled", True)),
            magnitude_threshold=float(motion_raw.get("magnitude_threshold", 1.0)),
            magnitude_clip=float(motion_raw.get("magnitude_clip", 5.0)),
            farneback=FarnebackConfig(
                pyr_scale=float(farneback_raw.get("pyr_scale", 0.5)),
                levels=int(farneback_raw.get("levels", 3)),
                winsize=int(farneback_raw.get("winsize", 15)),
                iterations=int(farneback_raw.get("iterations", 3)),
                poly_n=int(farneback_raw.get("poly_n", 5)),
                poly_sigma=float(farneback_raw.get("poly_sigma", 1.2)),
            ),
            mog2=MOG2Config(
                enabled_for_fixed_camera=bool(mog2_raw.get("enabled_for_fixed_camera", True)),
                history=int(mog2_raw.get("history", 300)),
                variance_threshold=float(mog2_raw.get("variance_threshold", 16.0)),
                learning_rate=float(mog2_raw.get("learning_rate", -1.0)),
            ),
        ),
        risk=RiskConfig(
            weights=weights,
            area_saturation_ratio=float(risk_raw.get("area_saturation_ratio", 0.15)),
            growth_saturation_ratio=float(risk_raw.get("growth_saturation_ratio", 0.5)),
            moderate_threshold=float(risk_raw.get("moderate_threshold", 0.3)),
            high_threshold=float(risk_raw.get("high_threshold", 0.6)),
            critical_threshold=float(risk_raw.get("critical_threshold", 0.8)),
        ),
        video=VideoConfig(
            output_codec=str(video_raw.get("output_codec", "mp4v")),
            fallback_fps=float(video_raw.get("fallback_fps", 25.0)),
            display_scale=float(video_raw.get("display_scale", 1.0)),
            draw_motion_vector=bool(video_raw.get("draw_motion_vector", True)),
            draw_track_trail=bool(video_raw.get("draw_track_trail", True)),
            trail_length=int(video_raw.get("trail_length", 20)),
        ),
        output=OutputConfig(
            report_dir=_resolve(root, output_raw["report_dir"]),
            default_video=_resolve(root, output_raw["default_video"]),
            frame_log=_resolve(root, output_raw["frame_log"]),
            event_log=_resolve(root, output_raw["event_log"]),
            summary=_resolve(root, output_raw["summary"]),
        ),
    )

    if config.tracker.max_age < 1 or config.tracker.minimum_hits < 1:
        raise ValueError("Tracker max_age and minimum_hits must be positive")
    if config.temporal.window_size < 2:
        raise ValueError("Temporal window_size must be at least 2")
    if not 1 <= config.temporal.fire_minimum_hits <= config.temporal.window_size:
        raise ValueError("fire_minimum_hits must fit inside temporal window")
    if not 1 <= config.temporal.smoke_minimum_hits <= config.temporal.window_size:
        raise ValueError("smoke_minimum_hits must fit inside temporal window")
    thresholds = (
        config.risk.moderate_threshold,
        config.risk.high_threshold,
        config.risk.critical_threshold,
    )
    if thresholds != tuple(sorted(thresholds)) or thresholds[0] < 0 or thresholds[-1] > 1:
        raise ValueError("Risk thresholds must be ordered within [0, 1]")
    if config.video.trail_length < 1:
        raise ValueError("trail_length must be positive")
    return config
