from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .models import Box


def class_signature(boxes: list[Box]) -> str:
    present = {box.class_id for box in boxes}
    if not present:
        return "negative"
    if present == {0}:
        return "fire_only"
    if present == {1}:
        return "smoke_only"
    if present == {0, 1}:
        return "fire_and_smoke"
    return "other"


def _validate_or_repair_box(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    repair: bool,
    min_box_pixels: int,
) -> tuple[Box | None, list[str]]:
    warnings: list[str] = []
    values = [x_center, y_center, width, height]
    if any(not isinstance(value, (int, float)) for value in values):
        return None, ["non_numeric_box"]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    out_of_bounds = x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1
    if out_of_bounds:
        if not repair:
            return None, ["out_of_bounds_box"]
        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        x2 = min(1.0, max(0.0, x2))
        y2 = min(1.0, max(0.0, y2))
        warnings.append("clamped_out_of_bounds_box")

    if x2 <= x1 or y2 <= y1:
        return None, warnings + ["zero_or_negative_area_box"]

    pixel_width = (x2 - x1) * image_width
    pixel_height = (y2 - y1) * image_height
    if pixel_width < min_box_pixels or pixel_height < min_box_pixels:
        return None, warnings + ["tiny_box_dropped"]

    box = Box(
        class_id=class_id,
        x_center=(x1 + x2) / 2,
        y_center=(y1 + y2) / 2,
        width=x2 - x1,
        height=y2 - y1,
    )
    return box, warnings


def parse_yolo(
    label_path: Path | None,
    class_map: dict[int, int],
    image_width: int,
    image_height: int,
    repair: bool,
    drop_unknown_classes: bool,
    min_box_pixels: int,
) -> tuple[list[Box], list[str], bool]:
    if label_path is None or not label_path.exists():
        return [], [], True

    boxes: list[Box] = []
    warnings: list[str] = []
    valid_file = True
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            warnings.append(f"line_{line_number}:expected_5_fields")
            valid_file = False
            continue
        try:
            source_class = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            warnings.append(f"line_{line_number}:invalid_number")
            valid_file = False
            continue

        if source_class not in class_map:
            warnings.append(f"line_{line_number}:unknown_class_{source_class}")
            if drop_unknown_classes:
                continue
            valid_file = False
            continue

        box, box_warnings = _validate_or_repair_box(
            class_id=class_map[source_class],
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            repair=repair,
            min_box_pixels=min_box_pixels,
        )
        warnings.extend(f"line_{line_number}:{warning}" for warning in box_warnings)
        if box is not None:
            boxes.append(box)
        elif "tiny_box_dropped" not in box_warnings:
            valid_file = False

    return boxes, warnings, valid_file


def parse_voc(
    label_path: Path | None,
    class_map: dict[str, int],
    image_width: int,
    image_height: int,
    repair: bool,
    drop_unknown_classes: bool,
    min_box_pixels: int,
) -> tuple[list[Box], list[str], bool]:
    if label_path is None or not label_path.exists():
        return [], [], True

    boxes: list[Box] = []
    warnings: list[str] = []
    valid_file = True
    try:
        root = ET.parse(label_path).getroot()
    except ET.ParseError:
        return [], ["invalid_xml"], False

    for object_index, obj in enumerate(root.findall("object"), start=1):
        name_node = obj.find("name")
        class_name = (name_node.text or "").strip().lower() if name_node is not None else ""
        if class_name not in class_map:
            warnings.append(f"object_{object_index}:unknown_class_{class_name or 'empty'}")
            if drop_unknown_classes:
                continue
            valid_file = False
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            warnings.append(f"object_{object_index}:missing_bndbox")
            valid_file = False
            continue
        try:
            xmin = float(bbox.findtext("xmin", "nan"))
            ymin = float(bbox.findtext("ymin", "nan"))
            xmax = float(bbox.findtext("xmax", "nan"))
            ymax = float(bbox.findtext("ymax", "nan"))
        except ValueError:
            warnings.append(f"object_{object_index}:invalid_coordinates")
            valid_file = False
            continue

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        box, box_warnings = _validate_or_repair_box(
            class_id=class_map[class_name],
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            repair=repair,
            min_box_pixels=min_box_pixels,
        )
        warnings.extend(f"object_{object_index}:{warning}" for warning in box_warnings)
        if box is not None:
            boxes.append(box)
        elif "tiny_box_dropped" not in box_warnings:
            valid_file = False

    return boxes, warnings, valid_file
