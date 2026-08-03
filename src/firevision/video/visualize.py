from __future__ import annotations

import cv2
import numpy as np

from .config import VideoConfig
from .models import FusedTrack, TrackObservation


STATE_COLOURS = {
    "POSSIBLE_FIRE": (0, 165, 255),
    "CONFIRMED_FIRE": (0, 0, 255),
    "POSSIBLE_SMOKE": (180, 180, 180),
    "CONFIRMED_SMOKE": (255, 180, 0),
}


def annotate_frame(
    frame: np.ndarray,
    fused_tracks: list[FusedTrack],
    raw_tracks: dict[int, TrackObservation],
    video_config: VideoConfig,
    fps: float,
) -> np.ndarray:
    output = frame.copy()
    for fused in fused_tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in fused.bbox)
        colour = STATE_COLOURS.get(fused.temporal.state, (255, 255, 255))
        thickness = 3 if fused.temporal.confirmed else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, thickness)
        line1 = f"#{fused.track_id} {fused.temporal.state}"
        line2 = f"conf {fused.temporal.smoothed_confidence:.2f} risk {fused.risk.level} {fused.risk.score:.2f}"
        _label(output, line1, x1, max(18, y1 - 26), colour)
        _label(output, line2, x1, max(38, y1 - 6), colour)

        raw = raw_tracks.get(fused.track_id)
        if raw and video_config.draw_track_trail and len(raw.center_history) > 1:
            points = raw.center_history[-video_config.trail_length :]
            for first, second in zip(points, points[1:]):
                cv2.line(output, first, second, colour, 2)
        if video_config.draw_motion_vector and (fused.motion.dominant_dx or fused.motion.dominant_dy):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            endpoint = (
                int(cx + fused.motion.dominant_dx * 5),
                int(cy + fused.motion.dominant_dy * 5),
            )
            cv2.arrowedLine(output, (cx, cy), endpoint, colour, 2, tipLength=0.25)

    confirmed = sum(track.temporal.confirmed for track in fused_tracks)
    cv2.putText(
        output,
        f"FPS {fps:.1f} | active {len(fused_tracks)} | confirmed {confirmed}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _label(frame: np.ndarray, text: str, x: int, y: int, colour: tuple[int, int, int]) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(frame, (x, y - height - baseline - 3), (x + width + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
