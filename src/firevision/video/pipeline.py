from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import FrameUndistorter
from .config import VideoFusionConfig
from .detector import Detector, UltralyticsDetector
from .events import EventRecorder
from .models import FusedTrack, TrackObservation
from .motion import MotionAnalyzer
from .risk import risk_score
from .temporal import TemporalVoting
from .tracking import SortTracker
from .visualize import annotate_frame


FRAME_FIELDS = [
    "frame_index", "timestamp_seconds", "track_id", "class_id", "label", "observed",
    "x1", "y1", "x2", "y2", "state", "confirmed", "smoothed_confidence",
    "persistence", "motion_score", "mean_motion", "moving_pixel_ratio",
    "upward_motion_ratio", "foreground_ratio", "area_ratio", "growth_ratio",
    "risk_score", "risk_level", "processing_latency_ms", "processing_fps",
]


class VideoFusionEngine:
    def __init__(
        self,
        config: VideoFusionConfig,
        detector: Detector,
        fixed_camera: bool = False,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = SortTracker(config.tracker)
        self.temporal = TemporalVoting(config.temporal)
        self.motion = MotionAnalyzer(config.motion, fixed_camera=fixed_camera)
        self.events = EventRecorder()
        self.previous_area: dict[int, float] = {}
        self.frame_rows: list[dict[str, Any]] = []

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
        measured_fps: float = 0.0,
    ) -> tuple[np.ndarray, list[FusedTrack]]:
        processing_started = time.perf_counter()
        detections = self.detector.infer(frame)
        raw_tracks = self.tracker.update(detections)
        self.motion.update_frame(frame)
        fused_tracks: list[FusedTrack] = []
        height, width = frame.shape[:2]
        frame_area = max(1.0, float(width * height))
        active_ids: set[int] = set()
        for track in raw_tracks:
            active_ids.add(track.track_id)
            temporal = self.temporal.update(track, frame_index)
            motion = self.motion.evidence(track.bbox, frame.shape)
            x1, y1, x2, y2 = track.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = area / frame_area
            previous = self.previous_area.get(track.track_id)
            growth_ratio = 0.0 if previous is None or previous <= 0 else (area - previous) / previous
            self.previous_area[track.track_id] = area
            risk = risk_score(self.config.risk, temporal, motion, area_ratio, growth_ratio)
            fused = FusedTrack(
                track_id=track.track_id,
                bbox=track.bbox,
                class_id=track.class_id,
                label=track.label,
                observed=track.observed,
                temporal=temporal,
                motion=motion,
                risk=risk,
                area_ratio=area_ratio,
                growth_ratio=growth_ratio,
            )
            fused_tracks.append(fused)
        processing_latency_ms = (time.perf_counter() - processing_started) * 1000.0
        processing_fps = 1000.0 / max(processing_latency_ms, 1e-6)
        for fused in fused_tracks:
            row = fused.as_row(frame_index, timestamp_seconds)
            row["processing_latency_ms"] = round(processing_latency_ms, 6)
            row["processing_fps"] = round(processing_fps, 6)
            self.frame_rows.append(row)
        self.temporal.prune(active_ids)
        for track_id in list(self.previous_area):
            if track_id not in active_ids:
                del self.previous_area[track_id]
        self.events.update(fused_tracks, frame_index, timestamp_seconds)
        raw_by_id = {track.track_id: track for track in raw_tracks}
        annotated = annotate_frame(frame, fused_tracks, raw_by_id, self.config.video, measured_fps)
        return annotated, fused_tracks

    def finish(self, frame_index: int, timestamp_seconds: float) -> None:
        self.events.finish_all(frame_index, timestamp_seconds)


def run_video(
    config: VideoFusionConfig,
    source: str | int,
    output_path: str | Path | None = None,
    display: bool = True,
    fixed_camera: bool = False,
    max_frames: int | None = None,
    detector: Detector | None = None,
) -> dict[str, Any]:
    resolved_source: str | int = int(source) if isinstance(source, str) and source.isdigit() else source
    capture = cv2.VideoCapture(resolved_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = config.video.fallback_fps
    destination = Path(output_path).expanduser().resolve() if output_path else config.output.default_video
    destination.parent.mkdir(parents=True, exist_ok=True)
    config.output.report_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*config.video.output_codec),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {destination}")

    engine = VideoFusionEngine(config, detector or UltralyticsDetector(config), fixed_camera=fixed_camera)
    undistorter = FrameUndistorter(config.camera.undistort, config.camera.calibration_file)
    frame_index = 0
    started = time.perf_counter()
    last_tick = started
    measured_fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.perf_counter()
            instantaneous = 1.0 / max(now - last_tick, 1e-6)
            measured_fps = instantaneous if measured_fps == 0 else 0.1 * instantaneous + 0.9 * measured_fps
            last_tick = now
            frame = undistorter.apply(frame)
            timestamp = frame_index / source_fps
            annotated, _ = engine.process_frame(frame, frame_index, timestamp, measured_fps)
            writer.write(annotated)
            if display:
                shown = annotated
                if config.video.display_scale != 1.0:
                    shown = cv2.resize(annotated, None, fx=config.video.display_scale, fy=config.video.display_scale)
                cv2.imshow("FireVision Video Fusion", shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    duration = frame_index / source_fps if source_fps > 0 else 0.0
    engine.finish(frame_index, duration)
    elapsed = time.perf_counter() - started
    _write_csv(config.output.frame_log, FRAME_FIELDS, engine.frame_rows)
    events = [event.as_dict() for event in engine.events.completed]
    event_fields = [
        "event_id", "track_id", "label", "start_frame", "start_seconds",
        "end_frame", "end_seconds", "peak_risk", "peak_confidence", "peak_risk_level",
    ]
    _write_csv(config.output.event_log, event_fields, events)
    counts = defaultdict(int)
    for event in events:
        counts[str(event["label"])] += 1
    summary = {
        "source": str(source),
        "output_video": str(destination),
        "frames_processed": frame_index,
        "source_fps": source_fps,
        "video_duration_seconds": duration,
        "wall_time_seconds": elapsed,
        "processing_fps": frame_index / max(elapsed, 1e-6),
        "fixed_camera_mode": fixed_camera,
        "camera_undistortion": config.camera.undistort,
        "events_total": len(events),
        "events_by_class": dict(counts),
        "frame_evidence_rows": len(engine.frame_rows),
        "frame_log": str(config.output.frame_log),
        "event_log": str(config.output.event_log),
    }
    config.output.summary.parent.mkdir(parents=True, exist_ok=True)
    config.output.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_report(config.output.report_dir / "REPORT.md", summary, events)
    return summary


def _write_markdown_report(path: Path, summary: dict[str, Any], events: list[dict[str, Any]]) -> None:
    lines = [
        "# Video Fusion Video Fusion Report",
        "",
        "## Run summary",
        "",
        f"- Source: `{summary['source']}`",
        f"- Frames processed: {summary['frames_processed']}",
        f"- Video duration: {summary['video_duration_seconds']:.3f} seconds",
        f"- Processing throughput: {summary['processing_fps']:.3f} FPS",
        f"- Fixed-camera MOG2 mode: {summary['fixed_camera_mode']}",
        f"- Confirmed events: {summary['events_total']}",
        f"- Events by class: {summary['events_by_class']}",
        "",
        "## Confirmed events",
        "",
    ]
    if not events:
        lines.append("No confirmed events were recorded.")
    else:
        lines.extend([
            "| Event | Track | Class | Start (s) | End (s) | Peak confidence | Peak risk | Level |",
            "|---:|---:|---|---:|---:|---:|---:|---|",
        ])
        for event in events:
            lines.append(
                "| {event_id} | {track_id} | {label} | {start_seconds:.3f} | {end_seconds:.3f} | "
                "{peak_confidence:.3f} | {peak_risk:.3f} | {peak_risk_level} |".format(**event)
            )
    lines.extend([
        "",
        "## Interpretation warning",
        "",
        "The risk value is a visual prioritisation score derived from confidence, persistence, motion, area, and growth. "
        "It is not a temperature, combustion-intensity, damage, or fire-spread estimate.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
