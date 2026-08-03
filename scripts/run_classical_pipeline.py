from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.classical.config import load_config
from firevision.classical.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Classical ML classical CV and SVM experiments")
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    summary = run_pipeline(load_config(PROJECT_ROOT / args.config))
    compact = {
        "patch_counts": summary["patch_counts"],
        "classical_test_macro_f1": summary["classical_test"]["macro_f1"],
        "svm_test_macro_f1": summary["svm_test"]["macro_f1"],
        "svm_best_parameters": summary["svm_training"]["best_parameters"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
