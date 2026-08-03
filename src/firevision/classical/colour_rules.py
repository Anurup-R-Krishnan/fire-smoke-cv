from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ColourConfig
from .contours import RegionStatistics, mask_region_statistics
from .dataset import PatchRecord
from .morphology import clean_binary_mask


@dataclass(frozen=True, slots=True)
class FirePixelThresholds:
    red_minus_green_min: float
    green_minus_blue_min: float
    saturation_min: float
    value_min: float
    cr_min: float
    cb_max: float
    hue_max: float


@dataclass(frozen=True, slots=True)
class SmokePixelThresholds:
    saturation_max: float
    value_min: float
    value_max: float
    cr_deviation_max: float
    cb_deviation_max: float
    red_green_difference_max: float
    green_blue_difference_max: float


@dataclass(frozen=True, slots=True)
class ColourThresholdModel:
    fire: FirePixelThresholds
    smoke: SmokePixelThresholds
    fire_vote_threshold: int
    smoke_vote_threshold: int
    fire_area_threshold: float = 0.02
    smoke_area_threshold: float = 0.04

    def to_dict(self) -> dict[str, object]:
        return {
            "fire": asdict(self.fire),
            "smoke": asdict(self.smoke),
            "fire_vote_threshold": self.fire_vote_threshold,
            "smoke_vote_threshold": self.smoke_vote_threshold,
            "fire_area_threshold": self.fire_area_threshold,
            "smoke_area_threshold": self.smoke_area_threshold,
        }

    def with_area_thresholds(self, fire: float, smoke: float) -> "ColourThresholdModel":
        return ColourThresholdModel(
            fire=self.fire,
            smoke=self.smoke,
            fire_vote_threshold=self.fire_vote_threshold,
            smoke_vote_threshold=self.smoke_vote_threshold,
            fire_area_threshold=float(fire),
            smoke_area_threshold=float(smoke),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ColourThresholdModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            fire=FirePixelThresholds(**data["fire"]),
            smoke=SmokePixelThresholds(**data["smoke"]),
            fire_vote_threshold=int(data["fire_vote_threshold"]),
            smoke_vote_threshold=int(data["smoke_vote_threshold"]),
            fire_area_threshold=float(data["fire_area_threshold"]),
            smoke_area_threshold=float(data["smoke_area_threshold"]),
        )


@dataclass(frozen=True, slots=True)
class ClassicalPrediction:
    predicted_class: int
    fire_score: float
    smoke_score: float
    fire_statistics: RegionStatistics
    smoke_statistics: RegionStatistics
    fire_mask: np.ndarray
    smoke_mask: np.ndarray


def _sample_class_pixels(
    records: list[PatchRecord],
    class_id: int,
    config: ColourConfig,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + class_id * 4099)
    chunks: list[np.ndarray] = []
    total = 0
    for record in records:
        if record.split != "train" or record.class_id != class_id:
            continue
        image = cv2.imread(str(record.patch_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        # Exclude black letterbox padding. This is crucial because the positive crop
        # may have a non-square aspect ratio.
        pixels = image.reshape(-1, 3)
        valid = np.any(pixels > 5, axis=1)
        pixels = pixels[valid]
        if not len(pixels):
            continue
        take = min(config.pixel_samples_per_image, len(pixels))
        indices = rng.choice(len(pixels), size=take, replace=False)
        chunks.append(pixels[indices])
        total += take
        if total >= config.maximum_pixels_per_class:
            break
    if not chunks:
        raise ValueError(f"No training pixels found for class {class_id}")
    combined = np.concatenate(chunks, axis=0)
    if len(combined) > config.maximum_pixels_per_class:
        indices = rng.choice(
            len(combined), size=config.maximum_pixels_per_class, replace=False
        )
        combined = combined[indices]
    return combined.astype(np.uint8, copy=False)


def _pixel_channels(bgr_pixels: np.ndarray) -> dict[str, np.ndarray]:
    image = bgr_pixels.reshape(-1, 1, 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    blue = bgr_pixels[:, 0].astype(np.float32)
    green = bgr_pixels[:, 1].astype(np.float32)
    red = bgr_pixels[:, 2].astype(np.float32)
    return {
        "blue": blue,
        "green": green,
        "red": red,
        "hue": hsv[:, 0].astype(np.float32),
        "saturation": hsv[:, 1].astype(np.float32),
        "value": hsv[:, 2].astype(np.float32),
        "cr": ycrcb[:, 1].astype(np.float32),
        "cb": ycrcb[:, 2].astype(np.float32),
    }


def fit_pixel_thresholds(
    records: list[PatchRecord],
    config: ColourConfig,
    seed: int,
) -> ColourThresholdModel:
    fire = _pixel_channels(_sample_class_pixels(records, 0, config, seed))
    smoke = _pixel_channels(_sample_class_pixels(records, 1, config, seed))

    # Quantiles are estimated only from training patches. Conservative domain
    # clamps keep background pixels inside loose bounding boxes from making the
    # rule meaningless.
    fire_thresholds = FirePixelThresholds(
        red_minus_green_min=float(max(0.0, np.quantile(fire["red"] - fire["green"], 0.25))),
        green_minus_blue_min=float(max(-5.0, np.quantile(fire["green"] - fire["blue"], 0.20))),
        saturation_min=float(max(35.0, np.quantile(fire["saturation"], 0.20))),
        value_min=float(max(70.0, np.quantile(fire["value"], 0.15))),
        cr_min=float(max(135.0, np.quantile(fire["cr"], 0.20))),
        cb_max=float(min(145.0, np.quantile(fire["cb"], 0.80))),
        hue_max=float(min(60.0, max(15.0, np.quantile(fire["hue"], 0.75)))),
    )

    smoke_thresholds = SmokePixelThresholds(
        saturation_max=float(min(125.0, max(35.0, np.quantile(smoke["saturation"], 0.80)))),
        value_min=float(max(35.0, np.quantile(smoke["value"], 0.08))),
        value_max=float(min(255.0, np.quantile(smoke["value"], 0.98))),
        cr_deviation_max=float(
            min(55.0, max(12.0, np.quantile(np.abs(smoke["cr"] - 128.0), 0.85)))
        ),
        cb_deviation_max=float(
            min(55.0, max(12.0, np.quantile(np.abs(smoke["cb"] - 128.0), 0.85)))
        ),
        red_green_difference_max=float(
            min(65.0, max(12.0, np.quantile(np.abs(smoke["red"] - smoke["green"]), 0.85)))
        ),
        green_blue_difference_max=float(
            min(65.0, max(12.0, np.quantile(np.abs(smoke["green"] - smoke["blue"]), 0.85)))
        ),
    )
    return ColourThresholdModel(
        fire=fire_thresholds,
        smoke=smoke_thresholds,
        fire_vote_threshold=config.fire_vote_threshold,
        smoke_vote_threshold=config.smoke_vote_threshold,
    )


def generate_colour_masks(
    bgr_image: np.ndarray,
    model: ColourThresholdModel,
    config: ColourConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if bgr_image is None or bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    blurred = cv2.GaussianBlur(
        bgr_image,
        (config.gaussian_kernel, config.gaussian_kernel),
        0,
    )
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    blue, green, red = cv2.split(blurred.astype(np.float32))
    hue, saturation, value = [channel.astype(np.float32) for channel in cv2.split(hsv)]
    _, cr, cb = [channel.astype(np.float32) for channel in cv2.split(ycrcb)]

    fire = model.fire
    fire_votes = np.stack(
        [
            red - green >= fire.red_minus_green_min,
            green - blue >= fire.green_minus_blue_min,
            saturation >= fire.saturation_min,
            value >= fire.value_min,
            cr >= fire.cr_min,
            cb <= fire.cb_max,
            (hue <= fire.hue_max) | (hue >= 170.0),
        ],
        axis=0,
    ).sum(axis=0)
    fire_mask = np.where(fire_votes >= model.fire_vote_threshold, 255, 0).astype(np.uint8)

    smoke = model.smoke
    smoke_votes = np.stack(
        [
            saturation <= smoke.saturation_max,
            value >= smoke.value_min,
            value <= smoke.value_max,
            np.abs(cr - 128.0) <= smoke.cr_deviation_max,
            np.abs(cb - 128.0) <= smoke.cb_deviation_max,
            np.abs(red - green) <= smoke.red_green_difference_max,
            np.abs(green - blue) <= smoke.green_blue_difference_max,
        ],
        axis=0,
    ).sum(axis=0)
    smoke_mask = np.where(smoke_votes >= model.smoke_vote_threshold, 255, 0).astype(np.uint8)
    smoke_mask[fire_mask > 0] = 0

    fire_mask = clean_binary_mask(
        fire_mask,
        config.morphology_kernel,
        config.opening_iterations,
        config.closing_iterations,
    )
    smoke_mask = clean_binary_mask(
        smoke_mask,
        config.morphology_kernel,
        config.opening_iterations,
        config.closing_iterations,
    )
    return fire_mask, smoke_mask


def classify_patch(
    bgr_image: np.ndarray,
    model: ColourThresholdModel,
    config: ColourConfig,
) -> ClassicalPrediction:
    fire_mask, smoke_mask = generate_colour_masks(bgr_image, model, config)
    fire_statistics = mask_region_statistics(fire_mask)
    smoke_statistics = mask_region_statistics(smoke_mask)
    fire_score = fire_statistics.area_ratio / max(model.fire_area_threshold, 1e-8)
    smoke_score = smoke_statistics.area_ratio / max(model.smoke_area_threshold, 1e-8)
    fire_positive = fire_statistics.area_ratio >= model.fire_area_threshold
    smoke_positive = smoke_statistics.area_ratio >= model.smoke_area_threshold
    if fire_positive or smoke_positive:
        predicted_class = 0 if fire_score >= smoke_score else 1
    else:
        predicted_class = 2
    return ClassicalPrediction(
        predicted_class=predicted_class,
        fire_score=float(fire_score),
        smoke_score=float(smoke_score),
        fire_statistics=fire_statistics,
        smoke_statistics=smoke_statistics,
        fire_mask=fire_mask,
        smoke_mask=smoke_mask,
    )
