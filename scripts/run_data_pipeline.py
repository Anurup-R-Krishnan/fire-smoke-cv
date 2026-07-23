#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.data.config import load_config
from firevision.data.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate, split, and standardize fire/smoke datasets."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "data.yaml"),
        help="Path to the Data Prep YAML configuration.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        summary = run_pipeline(config)
    except Exception as exc:  # CLI boundary: report a useful error without a stack trace.
        print(f"Data Prep failed: {exc}", file=sys.stderr)
        return 1

    print("\nData Prep completed successfully.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
