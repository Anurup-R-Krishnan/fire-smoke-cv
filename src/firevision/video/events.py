from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import FusedTrack


@dataclass(slots=True)
class Event:
    event_id: int
    track_id: int
    label: str
    start_frame: int
    start_seconds: float
    end_frame: int | None = None
    end_seconds: float | None = None
    peak_risk: float = 0.0
    peak_confidence: float = 0.0
    peak_risk_level: str = "LOW"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EventRecorder:
    def __init__(self) -> None:
        self._next_id = 1
        self._active: dict[int, Event] = {}
        self.completed: list[Event] = []

    def update(self, tracks: list[FusedTrack], frame_index: int, timestamp_seconds: float) -> None:
        confirmed_ids = {track.track_id for track in tracks if track.temporal.confirmed}
        by_id = {track.track_id: track for track in tracks}
        for track_id in confirmed_ids:
            track = by_id[track_id]
            event = self._active.get(track_id)
            if event is None:
                event = Event(
                    event_id=self._next_id,
                    track_id=track.track_id,
                    label=track.label,
                    start_frame=frame_index,
                    start_seconds=timestamp_seconds,
                )
                self._next_id += 1
                self._active[track_id] = event
            if track.risk.score >= event.peak_risk:
                event.peak_risk = track.risk.score
                event.peak_risk_level = track.risk.level
            event.peak_confidence = max(event.peak_confidence, track.temporal.smoothed_confidence)

        for track_id in list(self._active):
            if track_id not in confirmed_ids:
                self._finish(track_id, frame_index, timestamp_seconds)

    def finish_all(self, frame_index: int, timestamp_seconds: float) -> None:
        for track_id in list(self._active):
            self._finish(track_id, frame_index, timestamp_seconds)

    def _finish(self, track_id: int, frame_index: int, timestamp_seconds: float) -> None:
        event = self._active.pop(track_id)
        event.end_frame = frame_index
        event.end_seconds = timestamp_seconds
        self.completed.append(event)
