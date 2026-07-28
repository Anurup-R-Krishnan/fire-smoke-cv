from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from firevision.classical.config import load_config
from firevision.classical.dataset import prepare_patch_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fire/smoke/normal patches from Data Prep data")
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    records = prepare_patch_dataset(config)
    for split in ("train", "val", "test"):
        counts = Counter(record.class_name for record in records if record.split == split)
        print(split, dict(counts))
    print(f"Manifest: {config.output.patch_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
