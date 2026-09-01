#!/usr/bin/env python3
"""Predict the fire/smoke class of a single image using trained classical ML models.

Usage
-----
    uv run scripts/predict_image.py path/to/image.jpg

The script loads the best-selected models (SVM by default, with fallback to RF, ET,
XGB, LGBM if available) from artifacts/classical/, extracts HOG+LBP+colour+contour
features from the image, and prints the predicted class.

Output class labels
-------------------
    fire   — fire detected in the image
    smoke  — smoke detected in the image
    normal — no fire or smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import joblib
import numpy as np

from firevision.classical.config import load_config
from firevision.classical.dataset import CLASS_NAMES
from firevision.classical.features import extract_handcrafted_features


_MODEL_FILES = [
    ("SVM",          "hog_lbp_colour_rbf_svm.joblib"),
    ("Random Forest","rf_model.joblib"),
    ("Extra Trees",  "et_model.joblib"),
    ("XGBoost",      "xgb_model.joblib"),
    ("LightGBM",     "lgbm_model.joblib"),
]


def _load_best_model(artifact_dir: Path):
    """Return the first available trained model pipeline."""
    for name, filename in _MODEL_FILES:
        path = artifact_dir / filename
        if path.exists():
            print(f"[predict] Using model: {name} ({path.name})")
            return name, joblib.load(path)
    raise FileNotFoundError(
        f"No trained model found in {artifact_dir}. "
        "Run the classical pipeline first:\n"
        "  uv run scripts/run_classical_pipeline.py"
    )


def predict_image(image_path: Path, config_path: Path | None = None) -> str:
    """Predict the class of a single image. Returns 'fire', 'smoke', or 'normal'."""
    config_path = config_path or PROJECT_ROOT / "configs" / "classical.yaml"
    config = load_config(config_path)
    artifact_dir = config.output.artifact_dir

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Extract features
    features = extract_handcrafted_features(image, config.features)
    X = features.reshape(1, -1)  # shape (1, n_features)

    # Load model and predict
    model_name, model = _load_best_model(artifact_dir)
    class_id = int(model.predict(X)[0])
    class_name = CLASS_NAMES[class_id]

    # Print probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            print("[predict] Class probabilities:")
            for cid, prob in sorted(enumerate(proba), key=lambda x: -x[1]):
                print(f"          {CLASS_NAMES[cid]:6s}: {prob:.4f}")
        except Exception:
            pass

    return class_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict fire/smoke class of a single image using trained classical ML models."
    )
    parser.add_argument("image", type=Path, help="Path to input image (JPG/PNG/BMP/WebP).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to classical.yaml config. Defaults to configs/classical.yaml.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 1

    try:
        predicted_class = predict_image(args.image, args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n[result] Image : {args.image}")
    print(f"[result] Predicted class : {predicted_class.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
