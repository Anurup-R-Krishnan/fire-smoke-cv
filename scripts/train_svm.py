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
from firevision.classical.svm_model import evaluate_svm, train_and_select_svm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Classical ML RBF-SVM")
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    model, training = train_and_select_svm(config)
    metrics, _, _, _ = evaluate_svm(model, config, "test")
    print(json.dumps({"training": training, "test": metrics}, indent=2))


if __name__ == "__main__":
    main()
