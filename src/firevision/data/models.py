from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_yolo_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass(slots=True)
class Sample:
    source: str
    image_path: Path
    label_path: Path | None
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""
    phash: int | None = None
    duplicate_group: str = ""
    class_signature: str = "negative"
    split: str = ""
    output_image: Path | None = None
    output_label: Path | None = None

    @property
    def fire_boxes(self) -> int:
        return sum(box.class_id == 0 for box in self.boxes)

    @property
    def smoke_boxes(self) -> int:
        return sum(box.class_id == 1 for box in self.boxes)

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_image": str(self.image_path),
            "source_label": str(self.label_path) if self.label_path else "",
            "split": self.split,
            "output_image": str(self.output_image) if self.output_image else "",
            "output_label": str(self.output_label) if self.output_label else "",
            "width": self.width,
            "height": self.height,
            "boxes": len(self.boxes),
            "fire_boxes": self.fire_boxes,
            "smoke_boxes": self.smoke_boxes,
            "class_signature": self.class_signature,
            "sha256": self.sha256,
            "phash_hex": f"{self.phash:016x}" if self.phash is not None else "",
            "duplicate_group": self.duplicate_group,
            "warnings": " | ".join(self.warnings),
        }
