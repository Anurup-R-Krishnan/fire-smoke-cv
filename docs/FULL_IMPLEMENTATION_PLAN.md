# Full Implementation Plan: Phases 1–5

## System boundary

The project detects two visual classes:

```text
0 fire
1 smoke
```

It produces frame detections and video-level events. It does **not** claim to measure temperature, determine physical fire severity, predict spread, or replace certified alarm systems.

## Recommended 12-week sequence

| Phase | Weeks | Main result |
|---|---:|---|
| 1. Data foundation | 1–2 | Clean leakage-resistant dataset and calibration result |
| 2. Classical baselines | 3–4 | Colour, morphology, motion, HOG/LBP/SVM results |
| 3. Deep learning | 5–7 | Lightweight classifier and object detector |
| 4. Temporal system | 8–9 | Stable video events and false-alarm suppression |
| 5. Evaluation/deployment | 10–12 | Cross-dataset evidence, ablations, ONNX demo, report |

---

# Data Prep: Data foundation and calibration

## Inputs

- D-Fire images and YOLO labels
- Optional DFS images and Pascal VOC XML labels
- Optional team-recorded negative images
- Checkerboard images from the deployment camera

## Algorithms and engineering work

1. Recursive image discovery
2. OpenCV image decoding
3. YOLO annotation parser
4. Pascal VOC XML parser and conversion
5. Normalized-coordinate validation
6. Boundary clamping for repairable boxes
7. Tiny-box rejection
8. SHA-256 exact duplicate detection
9. DCT perceptual hashing
10. BK-tree Hamming-neighbour search
11. Disjoint-set near-duplicate clustering
12. Duplicate-group-aware stratified splitting
13. Hard-link/copy dataset materialization
14. YOLO `data.yaml` generation
15. Checkerboard corner detection
16. Subpixel corner refinement
17. Camera intrinsic/distortion estimation
18. Reprojection-error calculation

## Outputs

- `data/processed/fire_smoke/data.yaml`
- `images/{train,val,test}`
- `labels/{train,val,test}`
- Manifest, rejection log, duplicate report, summary, preview
- Camera matrix and distortion coefficients

## Acceptance tests

- Every retained image decodes successfully
- Every output label has valid normalized coordinates
- No exact duplicate remains
- Every near-duplicate group belongs to one split only
- Split totals equal retained-image count
- Fire/smoke/negative distributions are present in each sufficiently large split
- At least 10 usable calibration views
- Reprojection error is reported

---

# Classical ML: Classical CV and conventional ML

## Planned package structure

```text
src/firevision/classical/
├── colour_rules.py
├── morphology.py
├── contours.py
├── background.py
├── optical_flow.py
├── features.py
├── svm_model.py
└── evaluate.py
scripts/
├── run_classical_baseline.py
├── extract_features.py
└── train_svm.py
```

## Experiment A: Colour-rule baseline

For each image/frame:

1. Convert BGR to RGB, HSV, and YCrCb
2. Build candidate masks using data-derived percentile ranges
3. Apply 3×3 or 5×5 Gaussian smoothing only to the classical branch
4. Apply morphological opening and closing
5. Extract connected components/contours
6. Remove regions by relative area, not a fixed pixel count
7. Compute area, aspect ratio, solidity, extent, perimeter, and centroid

Variants:

- RGB only
- HSV only
- HSV + YCrCb
- HSV + YCrCb + morphology
- HSV + YCrCb + motion verification

## Experiment B: Motion

- MOG2 for fixed-camera sequences
- Farneback dense optical flow for adjacent frames
- Region statistics: mean magnitude, variance, moving-pixel ratio, dominant direction, upward-motion ratio

MOG2 is optional for moving phone footage. Optical flow remains supporting evidence rather than a hard rejection rule.

## Experiment C: HOG/LBP/colour + SVM

Features:

- HSV histogram
- YCrCb histogram
- HOG
- Uniform LBP histogram
- Optional contour statistics

Classifier:

- StandardScaler
- RBF-SVM
- Class weighting
- Three classes for patch classification: fire, smoke, normal

## Outputs

- Saved thresholds and SVM model
- Per-class confusion matrix
- Accuracy, macro precision, macro recall, macro F1
- False-positive category table
- Classical video examples

## Acceptance tests

- Deterministic feature extraction
- No test images used to tune thresholds
- At least one meaningful classical baseline
- Documented cases where classical methods fail

---

# Detector Training: Deep-learning classification and detection — implemented

## Implemented package structure

```text
src/firevision/detector/
├── classifier_data.py
├── classifier_models.py
├── classifier_train.py
├── config.py
├── detector.py
├── gradcam.py
├── metrics.py
├── pipeline.py
└── runtime.py
configs/detector.yaml
scripts/train_classifiers.py
scripts/train_detector.py
scripts/tune_detector_thresholds.py
scripts/generate_gradcam.py
scripts/run_detector.py
```

## Implemented experiments

- MobileNetV3-Small transfer learning for fire/smoke/normal patches
- Two-stage frozen-head and partial fine-tuning workflow
- Optional VGG16 syllabus comparison, disabled by default for laptop safety
- YOLO11n detector trials at 512 and 640 pixels
- Automatic mixed precision and conservative RTX 3050 Ti batch sizes
- Validation-only input-size selection using mAP50-95
- Validation-only confidence and NMS-IoU tuning
- One final detector test evaluation after configuration is frozen
- Grad-CAM failure overlays for the classifier
- Confusion matrices, learning curves, predictions, and experiment summaries

## Exit criterion

- `artifacts/detector/best_fire_smoke_detector.pt` exists
- `artifacts/detector/detector_thresholds.json` exists
- MobileNetV3 test and failure-analysis reports exist
- Test results are not used to choose architecture, image size, or thresholds

---

# Video Fusion: Temporal fusion and video events — implemented

## Implemented package structure

```text
src/firevision/video/
├── config.py
├── models.py
├── detector.py
├── tracking.py
├── temporal.py
├── motion.py
├── risk.py
├── events.py
├── visualize.py
└── pipeline.py
scripts/
├── check_video_environment.py
└── run_video.py
```

## Detection tracking

- Class-aware SORT-style tracking
- Constant-velocity Kalman state over box centre, width, height, and their velocities
- Hungarian assignment over IoU cost
- Configurable maximum missed-frame age
- Configurable minimum track hits

## Temporal confidence

Exponential moving average:

```text
smoothed = alpha × current + (1 - alpha) × previous
```

Default alpha: 0.4. It must be tuned using validation videos in Phase 5.

## N-of-M event confirmation

Starting rules:

```text
fire:  at least 3 observed detections in an 8-frame window
smoke: at least 4 observed detections in an 8-frame window
```

These are starting values, not final claims. Phase 5 freezes them after validation-video tuning.

## Motion fusion

Per tracked region:

- Farneback optical-flow magnitude
- Moving-pixel ratio
- Mean motion vector
- Upward-motion ratio
- Optional MOG2 overlap for fixed-camera video

Motion adjusts the visual risk score but is not a mandatory condition for confirming a persistent detector event.

## Visual risk indicator

The bounded score combines:

- Smoothed detector confidence
- Temporal persistence
- Region/frame area ratio
- Positive region growth rate
- Motion evidence

The report must call this a **visual risk indicator**, not physical severity.

## Event states

```text
POSSIBLE_SMOKE
POSSIBLE_FIRE
CONFIRMED_SMOKE
CONFIRMED_FIRE
```

Confirmed event intervals are written to a separate log so one persistent event does not become one alert per frame.

## Outputs

- Annotated MP4 video
- Per-frame evidence CSV
- Confirmed-event CSV
- JSON run summary
- Markdown run report
- Track trails, motion vectors, risk level, and measured FPS overlay

## Acceptance tests

- One-frame false detections do not trigger confirmed events
- Track IDs persist across adjacent detections
- Tracks survive short detector misses
- Event timestamps are deterministic for the same input
- Moving-camera mode does not use MOG2
- Video, CSV, JSON, and Markdown outputs are created

See `docs/PHASE4_IMPLEMENTATION.md` for execution instructions.

---

# Phase 5: Evaluation, deployment, and report

## Planned package structure

```text
src/firevision/phase5/
├── metrics_detection.py
├── metrics_video.py
├── cross_dataset.py
├── ablation.py
├── benchmark.py
├── export_onnx.py
└── report_tables.py
scripts/
├── evaluate_all.py
├── benchmark_runtime.py
├── export_model.py
└── demo.py
```

## Evaluation datasets

1. Internal held-out test split
2. MIVIA fire videos
3. MIVIA smoke videos
4. Team-recorded safe negative videos
5. Team-recorded screen/playback or controlled public video tests, clearly labelled as such

Never create uncontrolled fires for data collection.

## Image metrics

- Precision
- Recall
- F1
- AP50
- AP50-95
- Per-class AP

## Video/event metrics

- Frame recall
- Event recall
- False confirmed events per minute
- Time to first confirmed detection
- Event precision
- Average/median/95th-percentile latency
- FPS
- Peak GPU memory

## Required ablation

| Configuration | Purpose |
|---|---|
| Detector only | Frame baseline |
| + confidence EMA | Confidence stability |
| + N-of-M voting | Transient false-alarm suppression |
| + SORT/Kalman | Box/track continuity |
| + optical flow | Motion evidence |
| Full system | Final result |

## Cross-dataset study

Train only on Data Prep training data. Evaluate without fine-tuning on MIVIA/team data. This measures domain generalisation instead of memorisation.

## Deployment

- Export detector to ONNX
- Compare PyTorch and ONNX Runtime
- Use FP16 only when supported and verified
- Display class, confidence, track age, event state, and FPS
- Save event log and annotated output

## Final report structure

1. Abstract
2. Problem definition and scope
3. Literature/dataset review
4. Dataset preparation and leakage prevention
5. Classical CV methods
6. Deep-learning methods
7. Temporal fusion
8. Experimental protocol
9. Results
10. Ablation study
11. Cross-dataset evaluation
12. Runtime analysis
13. Failure cases
14. Ethical/safety limitations
15. Conclusion and future work

## Final acceptance criteria

- Reproducible commands for all reported results
- No train/test leakage
- Baseline-to-final comparison
- Cross-dataset test
- Runtime on the actual RTX 3050 Ti laptop
- Honest limitations and failure examples
- Demo works from a clean environment

