from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = {
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "matplotlib": "Matplotlib",
}


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failures = []
    for module_name, display_name in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
            print(f"{display_name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            failures.append(f"{display_name}: {exc}")
    if failures:
        raise SystemExit("Missing/broken dependencies:\n" + "\n".join(failures))
    print("Classical ML environment is ready.")


if __name__ == "__main__":
    main()
