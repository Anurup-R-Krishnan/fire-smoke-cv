from __future__ import annotations

import importlib
import platform
import sys


REQUIRED = [
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "numpy",
    "PIL",
    "sklearn",
    "matplotlib",
    "yaml",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failed = False
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"{name}: OK {getattr(module, '__version__', '')}")
        except Exception as exc:
            failed = True
            print(f"{name}: FAILED: {exc}")
    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            properties = torch.cuda.get_device_properties(0)
            print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")
        else:
            print(
                "WARNING: PyTorch is CPU-only or cannot access the NVIDIA driver. "
                "Classifier training will be slow and YOLO training is not recommended."
            )
    except Exception as exc:
        failed = True
        print(f"PyTorch CUDA check failed: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
