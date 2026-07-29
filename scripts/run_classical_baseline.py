from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.classical.colour_rules import fit_pixel_thresholds
from firevision.classical.config import load_config
from firevision.classical.dataset import load_patch_manifest
from firevision.classical.evaluate import evaluate_classical_records, tune_area_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the classical colour baseline")
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    records = load_patch_manifest(config)
    model = fit_pixel_thresholds(records, config.colour, config.seed)
    model, _ = tune_area_thresholds(records, model, config)
    model_path = config.output.artifact_dir / "colour_thresholds.json"
    model.save(model_path)
    metrics, _, _ = evaluate_classical_records(records, "test", model, config)
    print(json.dumps(metrics, indent=2))
    print(f"Saved thresholds: {model_path}")


if __name__ == "__main__":
    main()
