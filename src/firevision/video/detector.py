from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import VideoFusionConfig
from .models import Detection


class Detector(Protocol):
    def infer(self, frame: np.ndarray) -> list[Detection]: ...


class UltralyticsDetector:
    def __init__(self, config: VideoFusionConfig) -> None:
        if not config.model.checkpoint.exists():
            raise FileNotFoundError(
                f"Detector checkpoint not found: {config.model.checkpoint}. Run Detector Training first."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required. Install requirements-video.txt") from exc
        self.model = YOLO(str(config.model.checkpoint))
        self.device = _resolve_device(config.device)
        self.max_det = config.model.max_det
        self.image_size, self.confidence, self.iou = _load_thresholds(config)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        detections: list[Detection] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, classes):
            if class_id not in (0, 1):
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in bbox.tolist()),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    label="fire" if class_id == 0 else "smoke",
                )
            )
        return detections


def _load_thresholds(config: VideoFusionConfig) -> tuple[int, float, float]:
    image_size = config.model.fallback_image_size
    confidence = config.model.fallback_confidence
    iou = config.model.fallback_iou
    if config.model.thresholds.exists():
        raw = json.loads(config.model.thresholds.read_text(encoding="utf-8"))
        image_size = int(raw.get("selected_image_size", image_size))
        confidence = float(raw.get("selected_confidence", confidence))
        iou = float(raw.get("selected_iou", iou))
    return image_size, confidence, iou


def _resolve_device(value: str) -> str | int:
    if value != "auto":
        return int(value) if value.isdigit() else value
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
