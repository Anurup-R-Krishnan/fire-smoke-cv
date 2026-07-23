#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failures: list[str] = []

    try:
        import cv2

        print(f"OpenCV: {cv2.__version__}")
    except Exception as exc:
        failures.append(f"OpenCV import failed: {exc}")

    try:
        import numpy as np

        print(f"NumPy: {np.__version__}")
    except Exception as exc:
        failures.append(f"NumPy import failed: {exc}")

    try:
        import yaml

        print(f"PyYAML: {yaml.__version__}")
    except Exception as exc:
        failures.append(f"PyYAML import failed: {exc}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU memory: {memory:.2f} GiB")
    except ImportError:
        print("PyTorch: not installed yet (expected during Data Prep)")
    except Exception as exc:
        failures.append(f"PyTorch check failed: {exc}")

    if failures:
        print("\nEnvironment problems:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nEnvironment is ready for Data Prep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
