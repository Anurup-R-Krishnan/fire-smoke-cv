from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import TemporalConfig
from .models import TemporalEvidence, TrackObservation


@dataclass(slots=True)
class _State:
    observations: deque[int]
    smoothed_confidence: float = 0.0
    initialised: bool = False
    last_seen_frame: int = -1


class TemporalVoting:
    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self._states: dict[int, _State] = {}

    def update(self, track: TrackObservation, frame_index: int) -> TemporalEvidence:
        state = self._states.setdefault(
            track.track_id,
            _State(observations=deque(maxlen=self.config.window_size)),
        )
        observed = int(track.observed)
        state.observations.append(observed)
        if track.observed:
            if not state.initialised:
                state.smoothed_confidence = track.confidence
                state.initialised = True
            else:
                alpha = self.config.confidence_alpha
                state.smoothed_confidence = alpha * track.confidence + (1.0 - alpha) * state.smoothed_confidence
            state.last_seen_frame = frame_index

        hits = int(sum(state.observations))
        window_length = len(state.observations)
        persistence = hits / max(1, window_length)
        required_hits = (
            self.config.fire_minimum_hits
            if track.label == "fire"
            else self.config.smoke_minimum_hits
        )
        confirmed = (
            hits >= required_hits
            and state.smoothed_confidence >= self.config.minimum_confirmed_confidence
        )
        prefix = "FIRE" if track.label == "fire" else "SMOKE"
        status = f"CONFIRMED_{prefix}" if confirmed else f"POSSIBLE_{prefix}"
        return TemporalEvidence(
            state=status,
            observed_hits=hits,
            window_length=window_length,
            persistence=persistence,
            smoothed_confidence=state.smoothed_confidence,
            confirmed=confirmed,
        )

    def prune(self, active_track_ids: set[int]) -> None:
        stale = set(self._states) - active_track_ids
        for track_id in stale:
            del self._states[track_id]
