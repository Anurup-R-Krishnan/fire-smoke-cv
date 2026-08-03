#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture checkerboard images for camera calibration.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cols", type=int, default=9, help="Internal checkerboard corners horizontally")
    parser.add_argument("--rows", type=int, default=6, help="Internal checkerboard corners vertically")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", default="data/raw/calibration")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera index {args.camera}")
        return 1

    captured = 0
    last_capture = 0.0
    pattern = (args.cols, args.rows)
    print("Move the checkerboard through different angles and positions.")
    print("Press SPACE to save a frame with detected corners. Press Q to quit.")

    while captured < args.count:
        ok, frame = camera.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern, corners, found)
        cv2.putText(
            display,
            f"Captured {captured}/{args.count} | corners={'yes' if found else 'no'}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Checkerboard capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32 and found and time.time() - last_capture > 0.4:
            destination = output / f"calibration_{captured:03d}.jpg"
            cv2.imwrite(str(destination), frame)
            print(f"Saved {destination}")
            captured += 1
            last_capture = time.time()

    camera.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured} usable views.")
    return 0 if captured >= 10 else 2


if __name__ == "__main__":
    raise SystemExit(main())
