# Video Fusion: Temporal Video Fusion

Video Fusion converts the frame-level detector from Detector Training into a stable video event system. The implementation is deliberately lightweight enough for an RTX 3050 Ti and does not require another model to be trained.

## Implemented pipeline

```text
Video or webcam
    ↓
Optional calibrated-camera undistortion
    ↓
YOLO11n fire/smoke detections
    ↓
Class-aware Hungarian IoU association
    ↓
Constant-velocity Kalman box filter
    ↓
SORT-style persistent track IDs
    ↓
Farneback optical-flow evidence
    ↓
Optional MOG2 foreground evidence
    ↓
Exponential confidence smoothing
    ↓
Class-specific N-of-M temporal voting
    ↓
Visual risk indicator
    ↓
Annotated video + frame evidence + event log
```

## Why tracking is used

Fire does not need a permanent identity in the same way a person does. Tracking is used here to:

- connect detections across adjacent frames;
- smooth unstable bounding boxes;
- survive short detector misses;
- calculate persistence and region growth;
- prevent a one-frame detection from immediately becoming an alarm.

The implementation uses a constant-velocity Kalman filter over centre position, width and height. Association uses the Hungarian algorithm over a class-aware IoU cost matrix.

## Temporal confirmation

The default decision rules are:

- Fire: at least 3 observed detections in an 8-frame window.
- Smoke: at least 4 observed detections in an 8-frame window.
- Smoothed confidence must remain at least 0.25.

States are:

```text
POSSIBLE_FIRE
CONFIRMED_FIRE
POSSIBLE_SMOKE
CONFIRMED_SMOKE
```

These parameters are starting values. Phase 5 will compare alternatives on validation videos.

## Motion evidence

Farneback dense optical flow is calculated once per pair of frames. For each tracked box the system records:

- mean motion magnitude;
- proportion of moving pixels;
- upward-motion ratio;
- mean horizontal and vertical vector;
- optional MOG2 foreground ratio.

Motion is supporting evidence. Weak motion does not delete a high-confidence persistent detection.

Use `--fixed-camera` only for stationary webcam or CCTV footage. It enables MOG2. Do not enable it for handheld video.

## Visual risk indicator

The risk score is not a physical severity estimate. It is an interpretable visual indicator:

```text
0.35 × smoothed confidence
+ 0.20 × temporal persistence
+ 0.15 × motion evidence
+ 0.15 × relative region area
+ 0.15 × positive region growth
```

Levels:

```text
LOW       < 0.30
MODERATE  0.30–0.59
HIGH      0.60–0.79
CRITICAL  ≥ 0.80
```

Do not claim that this predicts temperature, combustion intensity, material damage, or future spread.

## Installation

```bash
source .venv/bin/activate
pip install -r requirements-video.txt
python scripts/check_video_environment.py
```

Detector Training must have produced:

```text
artifacts/detector/best_fire_smoke_detector.pt
artifacts/detector/detector_thresholds.json
```

## Optional camera undistortion

Data Prep creates `artifacts/calibration/camera_calibration.npz`. To apply it to footage from the same physical camera, set:

```yaml
camera:
  undistort: true
  calibration_file: artifacts/calibration/camera_calibration.npz
```

Leave it disabled for downloaded datasets or footage captured with another camera.

## Run on a video

```bash
python scripts/run_video.py \
  --config configs/video.yaml \
  --source path/to/fire_video.mp4 \
  --output reports/video/fire_video_annotated.mp4
```

## Run on a stationary webcam

```bash
python scripts/run_video.py \
  --config configs/video.yaml \
  --source 0 \
  --fixed-camera
```

Press `q` or Escape to stop.

## Headless execution

```bash
python scripts/run_video.py \
  --config configs/video.yaml \
  --source path/to/video.mp4 \
  --no-display
```

## Outputs

```text
reports/video/
├── annotated_video.mp4
├── frame_evidence.csv
├── events.csv
├── summary.json
└── REPORT.md
```

`frame_evidence.csv` contains one row per active track per frame. It includes box coordinates, temporal state, smoothed confidence, persistence, motion, area, growth, risk, processing latency, and processing FPS.

`events.csv` contains confirmed events rather than every frame. It records start/end time and peak risk/confidence.

## What to validate manually

Test at least these categories:

- visible flame;
- early or faint smoke;
- fire becoming larger;
- smoke becoming dispersed;
- sunset;
- bright orange lamp;
- welding;
- steam;
- cloud movement;
- video or monitor showing fire;
- brief detector false positive;
- one or two missed detector frames.

Video Fusion is complete when the annotated output remains stable and the event log contains meaningful events rather than frame-by-frame spam.

## Automated verification

Video Fusion has automated tests for:

- track-ID persistence and missed-frame prediction;
- fire temporal confirmation;
- risk-score bounds;
- confirmed-event interval creation;
- annotated video writing;
- frame/event CSV generation;
- Markdown and JSON reporting;
- Data Prep calibration-file compatibility.

Run:

```bash
PYTHONPATH=src pytest -q tests/test_video.py
```

The delivered repository completed all 12 Data Prep–4 tests when run in smaller groups in the packaging environment. Splitting the run avoids an environment-specific pytest shutdown delay after the tests have completed.
