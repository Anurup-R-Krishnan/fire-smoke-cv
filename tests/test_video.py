from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from firevision.video.camera import FrameUndistorter
from firevision.video.config import load_config
from firevision.video.models import Detection, MotionEvidence, TemporalEvidence, TrackObservation
from firevision.video.pipeline import VideoFusionEngine, run_video
from firevision.video.risk import risk_score
from firevision.video.temporal import TemporalVoting
from firevision.video.tracking import SortTracker


class SequenceDetector:
    def __init__(self, sequence: list[list[Detection]]) -> None:
        self.sequence = sequence
        self.index = 0

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if self.index >= len(self.sequence):
            return []
        result = self.sequence[self.index]
        self.index += 1
        return result


def _write_config(tmp_path: Path) -> Path:
    config = {
        "project_root": ".",
        "device": "cpu",
        "camera": {"undistort": False, "calibration_file": "artifacts/calibration/camera.npz"},
        "model": {
            "checkpoint": "artifacts/detector/best.pt",
            "thresholds": "artifacts/detector/thresholds.json",
            "fallback_image_size": 640,
            "fallback_confidence": 0.25,
            "fallback_iou": 0.45,
            "max_det": 100,
        },
        "tracker": {
            "minimum_iou": 0.2,
            "max_age": 3,
            "minimum_hits": 1,
            "process_noise": 0.01,
            "measurement_noise": 0.1,
            "initial_covariance": 10.0,
        },
        "temporal": {
            "window_size": 5,
            "fire_minimum_hits": 3,
            "smoke_minimum_hits": 4,
            "confidence_alpha": 0.4,
            "minimum_confirmed_confidence": 0.25,
        },
        "motion": {
            "enabled": True,
            "magnitude_threshold": 0.2,
            "magnitude_clip": 5.0,
            "farneback": {
                "pyr_scale": 0.5,
                "levels": 2,
                "winsize": 9,
                "iterations": 2,
                "poly_n": 5,
                "poly_sigma": 1.1,
            },
            "mog2": {
                "enabled_for_fixed_camera": True,
                "history": 20,
                "variance_threshold": 12.0,
                "learning_rate": -1.0,
            },
        },
        "risk": {
            "weights": {
                "confidence": 0.35,
                "persistence": 0.2,
                "motion": 0.15,
                "area": 0.15,
                "growth": 0.15,
            },
            "area_saturation_ratio": 0.15,
            "growth_saturation_ratio": 0.5,
            "moderate_threshold": 0.3,
            "high_threshold": 0.6,
            "critical_threshold": 0.8,
        },
        "video": {
            "output_codec": "mp4v",
            "fallback_fps": 10.0,
            "display_scale": 1.0,
            "draw_motion_vector": True,
            "draw_track_trail": True,
            "trail_length": 10,
        },
        "output": {
            "report_dir": "reports/video",
            "default_video": "reports/video/annotated.mp4",
            "frame_log": "reports/video/frames.csv",
            "event_log": "reports/video/events.csv",
            "summary": "reports/video/summary.json",
        },
    }
    path = tmp_path / "configs" / "video.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _detection(x: float, confidence: float = 0.8, label: str = "fire") -> Detection:
    return Detection(
        bbox=(x, 20.0, x + 30.0, 60.0),
        confidence=confidence,
        class_id=0 if label == "fire" else 1,
        label=label,
    )


def test_sort_tracker_keeps_identity(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    tracker = SortTracker(config.tracker)
    first = tracker.update([_detection(10)])
    second = tracker.update([_detection(13)])
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert second[0].observed is True
    predicted = tracker.update([])
    assert predicted[0].track_id == first[0].track_id
    assert predicted[0].observed is False


def test_temporal_voting_and_risk(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    voting = TemporalVoting(config.temporal)
    evidence = None
    for frame_index in range(3):
        evidence = voting.update(
            TrackObservation(
                track_id=7,
                bbox=(10, 10, 40, 50),
                class_id=0,
                label="fire",
                confidence=0.8,
                observed=True,
                age=frame_index + 1,
                hits=frame_index + 1,
                time_since_update=0,
            ),
            frame_index,
        )
    assert evidence is not None and evidence.confirmed
    risk = risk_score(
        config.risk,
        evidence,
        MotionEvidence(score=0.5),
        area_ratio=0.1,
        growth_ratio=0.2,
    )
    assert 0.0 <= risk.score <= 1.0
    assert risk.level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_video_fusion_engine_confirms_event(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    detector = SequenceDetector([[_detection(10)], [_detection(12)], [_detection(14)], [_detection(16)]])
    engine = VideoFusionEngine(config, detector, fixed_camera=False)
    states = []
    for index in range(4):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + index * 2, 20), (40 + index * 2, 60), (255, 255, 255), -1)
        annotated, fused = engine.process_frame(frame, index, index / 10)
        assert annotated.shape == frame.shape
        states.append(fused[0].temporal.state)
    engine.finish(4, 0.4)
    assert "CONFIRMED_FIRE" in states
    assert len(engine.events.completed) == 1
    assert engine.events.completed[0].label == "fire"


def test_run_video_writes_logs(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (120, 100))
    assert writer.isOpened()
    for index in range(5):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + index * 2, 20), (40 + index * 2, 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    detector = SequenceDetector([[_detection(10 + index * 2)] for index in range(5)])
    output = tmp_path / "annotated.mp4"
    summary = run_video(
        config,
        source=str(source),
        output_path=output,
        display=False,
        detector=detector,
    )
    assert summary["frames_processed"] == 5
    assert output.exists() and output.stat().st_size > 0
    assert config.output.frame_log.exists()
    assert config.output.event_log.exists()
    assert (config.output.report_dir / "REPORT.md").exists()
    with config.output.event_log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["label"] == "fire"


def test_camera_undistorter_loads_data_style_npz(tmp_path: Path) -> None:
    calibration = tmp_path / "camera.npz"
    np.savez(
        calibration,
        camera_matrix=np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        distortion=np.zeros((1, 5), dtype=np.float64),
    )
    undistorter = FrameUndistorter(True, calibration)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    assert undistorter.apply(frame).shape == frame.shape
