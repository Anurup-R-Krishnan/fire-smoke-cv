from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from firevision.detector.config import load_config
from firevision.detector.gradcam import generate_failure_gallery


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for classifier failures")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--maximum-images", type=int, default=16)
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    outputs = generate_failure_gallery(
        config,
        Path(args.checkpoint),
        Path(args.predictions),
        maximum_images=args.maximum_images,
    )
    print(f"Generated {len(outputs)} Grad-CAM overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
