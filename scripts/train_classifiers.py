from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

from firevision.detector.classifier_train import train_enabled_classifiers
from firevision.detector.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Detector Training transfer-learning classifiers")
    parser.add_argument("--config", default="configs/detector.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    results = train_enabled_classifiers(config)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
