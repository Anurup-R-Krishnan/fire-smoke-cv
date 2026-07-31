from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

from firevision.detector.config import load_config
from firevision.detector.detector import tune_detector_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune YOLO confidence and NMS IoU on validation")
    parser.add_argument("--config", default="configs/detector.yaml")
    args = parser.parse_args()
    summary = tune_detector_thresholds(load_config(PROJECT_ROOT / args.config))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
