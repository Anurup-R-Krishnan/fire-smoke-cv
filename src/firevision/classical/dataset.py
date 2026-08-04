from __future__ import annotations

import csv
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ClassicalMLConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "fire", 1: "smoke", 2: "normal"}


@dataclass(frozen=True, slots=True)
class DetectionBox:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class PatchCandidate:
    split: str
    class_id: int
    source_image: Path
    crop: tuple[int, int, int, int]
    source_box_index: int | None


@dataclass(frozen=True, slots=True)
class PatchRecord:
    split: str
    class_id: int
    class_name: str
    patch_path: Path
    source_image: Path
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    source_box_index: int | None

    def to_row(self, project_root: Path) -> dict[str, str | int]:
        def relative(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(project_root.resolve()))
            except ValueError:
                return str(path.resolve())

        return {
            "split": self.split,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "patch_path": relative(self.patch_path),
            "source_image": relative(self.source_image),
            "crop_x1": self.crop_x1,
            "crop_y1": self.crop_y1,
            "crop_x2": self.crop_x2,
            "crop_y2": self.crop_y2,
            "source_box_index": "" if self.source_box_index is None else self.source_box_index,
        }


def discover_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[DetectionBox]:
    if not label_path.exists():
        return []
    boxes: list[DetectionBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected five YOLO fields")
        class_id = int(parts[0])
        if class_id not in {0, 1}:
            continue
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        x1 = int(round((x_center - box_width / 2.0) * width))
        y1 = int(round((y_center - box_height / 2.0) * height))
        x2 = int(round((x_center + box_width / 2.0) * width))
        y2 = int(round((y_center + box_height / 2.0) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        boxes.append(DetectionBox(class_id, x1, y1, x2, y2))
    return boxes


def _box_overlap_score(a: tuple[int, int, int, int], b: DetectionBox) -> float:
    ax1, ay1, ax2, ay2 = a
    ix1 = max(ax1, b.x1)
    iy1 = max(ay1, b.y1)
    ix2 = min(ax2, b.x2)
    iy2 = min(ay2, b.y2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, b.width * b.height)
    iou = intersection / float(area_a + area_b - intersection)
    return max(iou, intersection / float(area_a), intersection / float(area_b))


def _context_crop(box: DetectionBox, width: int, height: int, context: float) -> tuple[int, int, int, int]:
    box_width = box.width
    box_height = box.height
    x1 = int(round(box.x1 - context * box_width))
    y1 = int(round(box.y1 - context * box_height))
    x2 = int(round(box.x2 + context * box_width))
    y2 = int(round(box.y2 + context * box_height))
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _normal_crop_candidates(
    image_path: Path,
    width: int,
    height: int,
    boxes: list[DetectionBox],
    count: int,
    minimum_fraction: float,
    maximum_fraction: float,
    maximum_iou: float,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    if count <= 0:
        return []
    name_seed = int.from_bytes(hashlib.sha256(str(image_path).encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed ^ name_seed)
    candidates: list[tuple[int, int, int, int]] = []
    shorter_side = min(width, height)
    attempts = max(20, count * 30)
    for _ in range(attempts):
        fraction = rng.uniform(minimum_fraction, maximum_fraction)
        side = max(24, int(round(shorter_side * fraction)))
        side = min(side, width, height)
        if side < 24:
            break
        x1 = rng.randint(0, max(0, width - side))
        y1 = rng.randint(0, max(0, height - side))
        crop = (x1, y1, x1 + side, y1 + side)
        if all(_box_overlap_score(crop, box) <= maximum_iou for box in boxes):
            candidates.append(crop)
            if len(candidates) >= count:
                break
    return candidates


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot letterbox an empty image")
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def _collect_candidates(config: ClassicalMLConfig, split: str) -> dict[int, list[PatchCandidate]]:
    image_root = config.dataset_dir / split / "images"
    label_root = config.dataset_dir / split / "labels"
    by_class: dict[int, list[PatchCandidate]] = {0: [], 1: [], 2: []}

    if not image_root.exists():
        return by_class

    for image_path in discover_images(image_root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        boxes = read_yolo_boxes(label_path, width, height)

        for box_index, box in enumerate(boxes):
            if box.width < config.patches.min_box_pixels or box.height < config.patches.min_box_pixels:
                continue
            crop = _context_crop(box, width, height, config.patches.context)
            if crop[2] - crop[0] < config.patches.min_box_pixels:
                continue
            if crop[3] - crop[1] < config.patches.min_box_pixels:
                continue
            by_class[box.class_id].append(
                PatchCandidate(split, box.class_id, image_path, crop, box_index)
            )

        normal_count = (
            config.patches.normal_per_negative_image
            if not boxes
            else config.patches.normal_per_positive_image
        )
        for crop in _normal_crop_candidates(
            image_path=image_path,
            width=width,
            height=height,
            boxes=boxes,
            count=normal_count,
            minimum_fraction=config.patches.normal_crop_min_fraction,
            maximum_fraction=config.patches.normal_crop_max_fraction,
            maximum_iou=config.patches.normal_max_iou,
            seed=config.seed,
        ):
            by_class[2].append(PatchCandidate(split, 2, image_path, crop, None))
    return by_class


def _select_candidates(
    candidates: dict[int, list[PatchCandidate]],
    split: str,
    config: ClassicalMLConfig,
) -> dict[int, list[PatchCandidate]]:
    selected: dict[int, list[PatchCandidate]] = {}
    caps = config.patches.max_per_class.get(split, {})
    for class_id, items in candidates.items():
        class_name = CLASS_NAMES[class_id]
        limit = int(caps.get(class_name, len(items)))
        local_seed = config.seed + class_id * 1009 + {"train": 1, "val": 2, "test": 3}[split]
        rng = random.Random(local_seed)
        ordered = list(items)
        rng.shuffle(ordered)
        selected[class_id] = ordered[:limit]
    return selected


def prepare_patch_dataset(config: ClassicalMLConfig) -> list[PatchRecord]:
    output_root = config.output.patch_dir
    if output_root.exists() and config.output.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[PatchRecord] = []
    for split in ("train", "val", "test"):
        candidates = _select_candidates(_collect_candidates(config, split), split, config)
        for class_id in (0, 1, 2):
            class_name = CLASS_NAMES[class_id]
            destination = output_root / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for index, candidate in enumerate(candidates[class_id]):
                image = cv2.imread(str(candidate.source_image), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                x1, y1, x2, y2 = candidate.crop
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                patch = _letterbox(crop, config.patches.size)
                digest = hashlib.sha1(
                    f"{candidate.source_image}|{candidate.crop}|{candidate.class_id}".encode("utf-8")
                ).hexdigest()[:12]
                patch_path = destination / f"{index:05d}_{digest}.jpg"
                success = cv2.imwrite(
                    str(patch_path),
                    patch,
                    [cv2.IMWRITE_JPEG_QUALITY, config.patches.jpeg_quality],
                )
                if not success:
                    raise OSError(f"Failed to write patch: {patch_path}")
                records.append(
                    PatchRecord(
                        split=split,
                        class_id=class_id,
                        class_name=class_name,
                        patch_path=patch_path,
                        source_image=candidate.source_image,
                        crop_x1=x1,
                        crop_y1=y1,
                        crop_x2=x2,
                        crop_y2=y2,
                        source_box_index=candidate.source_box_index,
                    )
                )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "class_id",
            "class_name",
            "patch_path",
            "source_image",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "source_box_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_row(config.project_root) for record in records)
    return records


def load_patch_manifest(config: ClassicalMLConfig) -> list[PatchRecord]:
    manifest_path = config.output.patch_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Patch manifest not found at {manifest_path}. Run patch preparation first."
        )
    records: list[PatchRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patch_path = Path(row["patch_path"])
            source_image = Path(row["source_image"])
            if not patch_path.is_absolute():
                patch_path = config.project_root / patch_path
            if not source_image.is_absolute():
                source_image = config.project_root / source_image
            records.append(
                PatchRecord(
                    split=row["split"],
                    class_id=int(row["class_id"]),
                    class_name=row["class_name"],
                    patch_path=patch_path.resolve(),
                    source_image=source_image.resolve(),
                    crop_x1=int(row["crop_x1"]),
                    crop_y1=int(row["crop_y1"]),
                    crop_x2=int(row["crop_x2"]),
                    crop_y2=int(row["crop_y2"]),
                    source_box_index=(
                        None if row["source_box_index"] == "" else int(row["source_box_index"])
                    ),
                )
            )
    return records
