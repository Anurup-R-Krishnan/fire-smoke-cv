#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import cv2
    import numpy
    import scipy

    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"SciPy: {scipy.__version__}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info(0)
            print(f"VRAM free/total: {free / 2**30:.2f}/{total / 2**30:.2f} GiB")
    except ImportError:
        print("PyTorch: not installed")
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics: not installed")


if __name__ == "__main__":
    main()
