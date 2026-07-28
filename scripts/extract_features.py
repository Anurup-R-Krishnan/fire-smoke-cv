from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.classical.config import load_config
from firevision.classical.dataset import load_patch_manifest
from firevision.classical.features import extract_split_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HOG, LBP and colour features")
    parser.add_argument("--config", default="configs/classical.yaml")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    records = load_patch_manifest(config)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        output = extract_split_features(records, split, config)
        print(output)


if __name__ == "__main__":
    main()
