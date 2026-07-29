from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.classical.colour_rules import ColourThresholdModel
from firevision.classical.config import load_config
from firevision.classical.video import ClassicalVideoProcessor


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the classical colour/motion baseline on video")
    parser.add_argument("--config", default="configs/classical.yaml")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--fixed-camera", action="store_true", help="Enable MOG2 foreground evidence")
    parser.add_argument("--output", help="Optional annotated MP4 output")
    parser.add_argument("--evidence-csv", help="Optional per-frame evidence CSV")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    model = ColourThresholdModel.load(config.output.artifact_dir / "colour_thresholds.json")
    processor = ClassicalVideoProcessor(config, model, args.fixed_camera)
    capture = cv2.VideoCapture(_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    writer = None
    csv_handle = None
    csv_writer = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, evidence, _, _ = processor.process(frame)
            if args.output and writer is None:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            if writer is not None:
                writer.write(annotated)
            if args.evidence_csv and csv_writer is None:
                evidence_path = Path(args.evidence_csv)
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                csv_handle = evidence_path.open("w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(
                    csv_handle,
                    fieldnames=[
                        "frame",
                        "fire_area_ratio",
                        "smoke_area_ratio",
                        "foreground_ratio",
                        "mean_flow",
                        "moving_pixel_ratio",
                        "upward_motion_ratio",
                    ],
                )
                csv_writer.writeheader()
            if csv_writer is not None:
                csv_writer.writerow({"frame": frame_index, **asdict(evidence)})
            if not args.no_display:
                cv2.imshow("Classical ML classical baseline", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_handle is not None:
            csv_handle.close()
        cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames")


if __name__ == "__main__":
    main()
