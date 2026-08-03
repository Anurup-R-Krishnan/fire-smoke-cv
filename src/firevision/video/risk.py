from __future__ import annotations

import numpy as np

from .config import RiskConfig
from .models import MotionEvidence, RiskEvidence, TemporalEvidence


def risk_score(
    config: RiskConfig,
    temporal: TemporalEvidence,
    motion: MotionEvidence,
    area_ratio: float,
    growth_ratio: float,
) -> RiskEvidence:
    confidence_component = float(np.clip(temporal.smoothed_confidence, 0.0, 1.0))
    persistence_component = float(np.clip(temporal.persistence, 0.0, 1.0))
    motion_component = float(np.clip(motion.score, 0.0, 1.0))
    area_component = float(np.clip(area_ratio / max(config.area_saturation_ratio, 1e-6), 0.0, 1.0))
    growth_component = float(
        np.clip(max(0.0, growth_ratio) / max(config.growth_saturation_ratio, 1e-6), 0.0, 1.0)
    )
    weights = config.weights
    score = (
        weights.confidence * confidence_component
        + weights.persistence * persistence_component
        + weights.motion * motion_component
        + weights.area * area_component
        + weights.growth * growth_component
    )
    score = float(np.clip(score, 0.0, 1.0))
    if score >= config.critical_threshold:
        level = "CRITICAL"
    elif score >= config.high_threshold:
        level = "HIGH"
    elif score >= config.moderate_threshold:
        level = "MODERATE"
    else:
        level = "LOW"
    return RiskEvidence(
        score=score,
        level=level,
        confidence_component=confidence_component,
        persistence_component=persistence_component,
        motion_component=motion_component,
        area_component=area_component,
        growth_component=growth_component,
    )
