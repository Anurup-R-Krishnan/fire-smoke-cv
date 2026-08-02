#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firevision.video import load_config, run_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Video Fusion temporal fire/smoke video fusion")
    parser.add_argument("--config", default="configs/video.yaml")
    parser.add_argument("--source", required=True, help="Video path, stream URL, or webcam index such as 0")
    parser.add_argument("--output", help="Annotated MP4 path")
    parser.add_argument("--fixed-camera", action="store_true", help="Enable MOG2 foreground evidence")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_video(
        config=config,
        source=args.source,
        output_path=args.output,
        display=not args.no_display,
        fixed_camera=args.fixed_camera,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
