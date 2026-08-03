#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a camera from checkerboard images.")
    parser.add_argument("--images", default="data/raw/calibration")
    parser.add_argument("--cols", type=int, default=9, help="Internal checkerboard corners horizontally")
    parser.add_argument("--rows", type=int, default=6, help="Internal checkerboard corners vertically")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--output", default="artifacts/calibration/camera_calibration.npz")
    args = parser.parse_args()

    images_dir = Path(args.images)
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ) if images_dir.exists() else []
    if not image_paths:
        print(f"No calibration images found in {images_dir}")
        return 1

    pattern = (args.cols, args.rows)
    object_template = np.zeros((args.rows * args.cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    object_template *= args.square_mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used: list[str] = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        used.append(path.name)

    if image_size is None or len(object_points) < 10:
        print(f"Need at least 10 usable checkerboard views; found {len(object_points)}")
        return 2

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    errors: list[float] = []
    for index, object_set in enumerate(object_points):
        projected, _ = cv2.projectPoints(
            object_set, rvecs[index], tvecs[index], camera_matrix, distortion
        )
        error = cv2.norm(image_points[index], projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=np.array(image_size),
        rms=np.array(rms),
        mean_reprojection_error=np.array(np.mean(errors)),
    )
    report = {
        "usable_images": len(used),
        "used_files": used,
        "image_size": image_size,
        "rms": float(rms),
        "mean_reprojection_error": float(np.mean(errors)),
        "camera_matrix": camera_matrix.tolist(),
        "distortion": distortion.tolist(),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
