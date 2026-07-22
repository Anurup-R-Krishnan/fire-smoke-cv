from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from .annotations import class_signature, parse_voc, parse_yolo
from .config import DataPrepConfig, SourceConfig
from .hashing import BKTree, DisjointSet, file_sha256, perceptual_hash
from .models import Box, Sample
from .splitting import assign_splits

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray | None:
    """Read an image robustly, including paths containing non-ASCII characters."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError):
        return None


def _matching_label(source: SourceConfig, image_path: Path) -> Path:
    relative = image_path.relative_to(source.images_dir)
    suffix = ".txt" if source.format == "yolo" else ".xml"
    return source.labels_dir / relative.with_suffix(suffix)


def _discover_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_sources(config: DataPrepConfig) -> tuple[list[Sample], list[dict[str, Any]]]:
    samples: list[Sample] = []
    rejected: list[dict[str, Any]] = []
    validation = config.validation
    min_width = int(validation.get("min_width", 32))
    min_height = int(validation.get("min_height", 32))
    repair = bool(validation.get("repair_out_of_bounds_boxes", True))
    drop_unknown = bool(validation.get("drop_unknown_classes", True))
    min_box_pixels = int(validation.get("drop_tiny_boxes_pixels", 4))

    found_any_source = False
    for source in config.sources:
        if not source.images_dir.exists():
            rejected.append(
                {
                    "source": source.name,
                    "image": str(source.images_dir),
                    "reason": "source_images_directory_missing",
                }
            )
            continue
        found_any_source = True
        images = _discover_images(source.images_dir)
        if not images:
            rejected.append(
                {
                    "source": source.name,
                    "image": str(source.images_dir),
                    "reason": "source_contains_no_supported_images",
                }
            )
            continue

        for image_path in tqdm(images, desc=f"Inspecting {source.name}", unit="image"):
            image = read_image(image_path)
            if image is None:
                rejected.append(
                    {"source": source.name, "image": str(image_path), "reason": "corrupt_or_unreadable_image"}
                )
                continue
            height, width = image.shape[:2]
            if width < min_width or height < min_height:
                rejected.append(
                    {
                        "source": source.name,
                        "image": str(image_path),
                        "reason": f"image_too_small:{width}x{height}",
                    }
                )
                continue

            label_path = _matching_label(source, image_path)
            if not label_path.exists() and not source.missing_label_is_negative:
                rejected.append(
                    {
                        "source": source.name,
                        "image": str(image_path),
                        "reason": "missing_annotation",
                    }
                )
                continue

            if source.format == "yolo":
                boxes, warnings, valid_file = parse_yolo(
                    label_path if label_path.exists() else None,
                    source.class_map,  # type: ignore[arg-type]
                    width,
                    height,
                    repair,
                    drop_unknown,
                    min_box_pixels,
                )
            else:
                boxes, warnings, valid_file = parse_voc(
                    label_path if label_path.exists() else None,
                    source.class_map,  # type: ignore[arg-type]
                    width,
                    height,
                    repair,
                    drop_unknown,
                    min_box_pixels,
                )

            if not valid_file and not boxes:
                rejected.append(
                    {
                        "source": source.name,
                        "image": str(image_path),
                        "reason": "annotation_invalid_no_usable_boxes",
                        "warnings": " | ".join(warnings),
                    }
                )
                continue

            sample = Sample(
                source=source.name,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                width=width,
                height=height,
                boxes=boxes,
                warnings=warnings,
            )
            sample.class_signature = class_signature(boxes)
            sample.sha256 = file_sha256(image_path)
            if bool(config.deduplication.get("perceptual_hash", True)):
                sample.phash = perceptual_hash(image)
            samples.append(sample)

    if not found_any_source:
        configured = "\n".join(f"- {source.name}: {source.images_dir}" for source in config.sources)
        raise FileNotFoundError(
            "No configured dataset source exists. Download/arrange the data first, then update "
            f"configs/data.yaml if needed. Configured locations:\n{configured}"
        )
    return samples, rejected


def remove_exact_duplicates(samples: list[Sample]) -> tuple[list[Sample], list[dict[str, Any]]]:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.sha256].append(sample)

    retained: list[Sample] = []
    records: list[dict[str, Any]] = []
    for sha256, group in by_hash.items():
        # Prefer the copy with the most usable annotations, then deterministic path order.
        group.sort(key=lambda item: (-len(item.boxes), str(item.image_path)))
        kept = group[0]
        retained.append(kept)
        kept_label_signature = tuple(
            sorted((box.class_id, round(box.x_center, 6), round(box.y_center, 6), round(box.width, 6), round(box.height, 6)) for box in kept.boxes)
        )
        for duplicate in group[1:]:
            duplicate_signature = tuple(
                sorted((box.class_id, round(box.x_center, 6), round(box.y_center, 6), round(box.width, 6), round(box.height, 6)) for box in duplicate.boxes)
            )
            records.append(
                {
                    "reason": "exact_duplicate",
                    "kept": str(kept.image_path),
                    "removed": str(duplicate.image_path),
                    "sha256": sha256,
                    "annotation_conflict": kept_label_signature != duplicate_signature,
                }
            )
    retained.sort(key=lambda item: (item.source, str(item.image_path)))
    return retained, records


def assign_duplicate_groups(samples: list[Sample], max_distance: int) -> list[dict[str, Any]]:
    if not samples:
        return []
    dsu = DisjointSet(len(samples))
    tree = BKTree()
    hash_to_indices: dict[int, list[int]] = defaultdict(list)

    for index, sample in enumerate(tqdm(samples, desc="Grouping near duplicates", unit="image")):
        if sample.phash is None:
            continue
        matches = tree.query(sample.phash, max_distance)
        for matched_hash in matches:
            for other_index in hash_to_indices[matched_hash]:
                dsu.union(index, other_index)
        if sample.phash not in hash_to_indices:
            tree.add(sample.phash)
        hash_to_indices[sample.phash].append(index)

    roots: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        roots[dsu.find(index)].append(index)

    records: list[dict[str, Any]] = []
    for group_number, indices in enumerate(sorted(roots.values(), key=lambda values: min(values))):
        group_id = f"dup_{group_number:06d}"
        for index in indices:
            samples[index].duplicate_group = group_id
        if len(indices) > 1:
            representative = samples[indices[0]]
            for index in indices[1:]:
                sample = samples[index]
                records.append(
                    {
                        "reason": "near_duplicate_grouped_not_removed",
                        "group": group_id,
                        "representative": str(representative.image_path),
                        "member": str(sample.image_path),
                        "representative_phash": f"{representative.phash:016x}" if representative.phash is not None else "",
                        "member_phash": f"{sample.phash:016x}" if sample.phash is not None else "",
                    }
                )
    return records


def _transfer_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return "symlink"
    if mode != "hardlink":
        raise ValueError(f"Unknown transfer mode: {mode}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def materialize_dataset(config: DataPrepConfig, samples: list[Sample]) -> Counter[str]:
    output_dir: Path = config.output["dataset_dir"]
    overwrite = bool(config.output.get("overwrite", False))
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset exists and overwrite=false: {output_dir}")
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    transfer_mode = str(config.output.get("transfer_mode", "hardlink")).lower()
    transfer_counts: Counter[str] = Counter()
    for sample in tqdm(samples, desc="Writing processed dataset", unit="image"):
        safe_source = "".join(char if char.isalnum() or char in "-_" else "_" for char in sample.source)
        filename = f"{safe_source}__{sample.image_path.stem}__{sample.sha256[:10]}{sample.image_path.suffix.lower()}"
        image_destination = output_dir / "images" / sample.split / filename
        label_destination = output_dir / "labels" / sample.split / f"{Path(filename).stem}.txt"
        transfer_counts[_transfer_file(sample.image_path, image_destination, transfer_mode)] += 1
        label_destination.write_text(
            "\n".join(box.as_yolo_line() for box in sample.boxes) + ("\n" if sample.boxes else ""),
            encoding="utf-8",
        )
        sample.output_image = image_destination
        sample.output_label = label_destination

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": config.classes,
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    return transfer_counts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_preview(samples: list[Sample], destination: Path, count: int, seed: int) -> None:
    if not samples or count <= 0:
        return
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(count, len(samples)))
    tile_width, tile_height = 320, 240
    columns = 4
    rows = (len(chosen) + columns - 1) // columns
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)

    for index, sample in enumerate(chosen):
        image = read_image(sample.image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = min(tile_width / width, (tile_height - 26) / height)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        offset_x = (tile_width - resized.shape[1]) // 2
        offset_y = 22 + ((tile_height - 22) - resized.shape[0]) // 2
        tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
        tile[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
        for box in sample.boxes:
            x1 = int((box.x_center - box.width / 2) * resized.shape[1]) + offset_x
            y1 = int((box.y_center - box.height / 2) * resized.shape[0]) + offset_y
            x2 = int((box.x_center + box.width / 2) * resized.shape[1]) + offset_x
            y2 = int((box.y_center + box.height / 2) * resized.shape[0]) + offset_y
            cv2.rectangle(tile, (x1, y1), (x2, y2), (40, 40, 40), 2)
            cv2.putText(
                tile,
                "fire" if box.class_id == 0 else "smoke",
                (x1, max(offset_y + 12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            tile,
            f"{sample.split} | {sample.class_signature}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        canvas[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile

    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def write_reports(
    config: DataPrepConfig,
    samples: list[Sample],
    rejected: list[dict[str, Any]],
    duplicate_records: list[dict[str, Any]],
    split_counts: dict[str, int],
    transfer_counts: Counter[str],
) -> dict[str, Any]:
    report_dir: Path = config.output["report_dir"]
    if report_dir.exists() and bool(config.output.get("overwrite", False)):
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [sample.to_manifest_row() for sample in samples]
    _write_csv(report_dir / "manifest.csv", manifest_rows)
    _write_csv(report_dir / "rejected.csv", rejected)
    _write_csv(report_dir / "duplicates.csv", duplicate_records)

    signature_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts = Counter()
    box_counts = Counter()
    warning_counts = Counter()
    for sample in samples:
        signature_by_split[sample.split][sample.class_signature] += 1
        source_counts[sample.source] += 1
        box_counts["fire"] += sample.fire_boxes
        box_counts["smoke"] += sample.smoke_boxes
        for warning in sample.warnings:
            warning_counts[warning.split(":")[-1]] += 1

    summary: dict[str, Any] = {
        "total_retained_images": len(samples),
        "total_rejected_records": len(rejected),
        "split_counts": split_counts,
        "class_signature_by_split": {
            split: dict(counts) for split, counts in signature_by_split.items()
        },
        "source_counts": dict(source_counts),
        "box_counts": dict(box_counts),
        "annotation_warning_counts": dict(warning_counts),
        "duplicate_records": len(duplicate_records),
        "transfer_counts": dict(transfer_counts),
        "dataset_directory": str(config.output["dataset_dir"]),
        "data_yaml": str(Path(config.output["dataset_dir"]) / "data.yaml"),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Data Prep Dataset Report",
        "",
        f"- Retained images: **{len(samples)}**",
        f"- Rejected records: **{len(rejected)}**",
        f"- Fire boxes: **{box_counts['fire']}**",
        f"- Smoke boxes: **{box_counts['smoke']}**",
        f"- Duplicate audit records: **{len(duplicate_records)}**",
        "",
        "## Split counts",
        "",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"- {split}: **{split_counts.get(split, 0)}**")
    lines.extend(["", "## Class signatures by split", ""])
    for split in ("train", "val", "test"):
        counts = signature_by_split[split]
        lines.append(
            f"- {split}: negative={counts['negative']}, fire-only={counts['fire_only']}, "
            f"smoke-only={counts['smoke_only']}, both={counts['fire_and_smoke']}"
        )
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `manifest.csv`: every retained sample and its final split",
            "- `rejected.csv`: unreadable images and invalid annotations",
            "- `duplicates.csv`: exact removals and near-duplicate groups",
            "- `preview.jpg`: visual label inspection montage",
            "- `summary.json`: machine-readable statistics",
            "",
            "Near duplicates are kept but assigned to one duplicate group, and the group is kept entirely inside one split. This prevents visually identical copies from leaking between training and evaluation.",
        ]
    )
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    create_preview(
        samples,
        report_dir / "preview.jpg",
        int(config.output.get("preview_samples", 24)),
        config.seed,
    )
    return summary


def run_pipeline(config: DataPrepConfig) -> dict[str, Any]:
    samples, rejected = inspect_sources(config)
    if not samples:
        raise ValueError("No valid images remained after source inspection and validation")
    duplicate_records: list[dict[str, Any]] = []
    if bool(config.deduplication.get("exact_duplicates", True)):
        samples, exact_records = remove_exact_duplicates(samples)
        duplicate_records.extend(exact_records)

    if bool(config.deduplication.get("perceptual_hash", True)):
        duplicate_records.extend(
            assign_duplicate_groups(
                samples, int(config.deduplication.get("phash_hamming_threshold", 4))
            )
        )
    else:
        for index, sample in enumerate(samples):
            sample.duplicate_group = f"sample_{index:06d}"

    split_counts = assign_splits(
        samples,
        float(config.split["train"]),
        float(config.split["val"]),
        float(config.split["test"]),
        config.seed,
    )
    transfer_counts = materialize_dataset(config, samples)
    return write_reports(
        config,
        samples,
        rejected,
        duplicate_records,
        split_counts,
        transfer_counts,
    )
