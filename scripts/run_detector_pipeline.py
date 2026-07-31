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
from firevision.detector.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Detector Training deep-learning experiments")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--skip-classifiers", action="store_true")
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-threshold-tuning", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(
        load_config(PROJECT_ROOT / args.config),
        run_classifiers=not args.skip_classifiers,
        run_detector=not args.skip_detector,
        run_threshold_tuning=not args.skip_threshold_tuning,
        run_gradcam=not args.skip_gradcam,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
