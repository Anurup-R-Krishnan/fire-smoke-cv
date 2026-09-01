## Real-Time Fire and Smoke Detection in Video with Temporal False-Alarm Suppression

The study compares classical computer vision, conventional machine learning, lightweight CNN classification, object detection, and temporal video processing. The final system detects fire/smoke, tracks persistent regions, and suppresses one-frame false alarms.

## Five implementation phases

### Data Prep: Data foundation and camera calibration — implemented

Deliverables:

- Reproducible repository and Python environment
- D-Fire/DFS ingestion
- YOLO and Pascal VOC annotation support
- Corrupt-image and annotation validation
- Bounding-box repair and tiny-box filtering
- SHA-256 exact deduplication
- DCT perceptual-hash near-duplicate grouping
- Leakage-resistant 70/15/15 split
- Standard YOLO dataset and `data.yaml`
- Dataset manifest, rejection log, duplicate audit, summary, and preview montage
- Checkerboard capture and camera-calibration scripts

Exit criterion: a clean dataset at `data/processed/fire_smoke/` and an auditable report at `reports/data/`.

### Classical ML: Classical CV and machine-learning baselines — implemented

Implemented:

1. RGB/HSV/YCrCb colour analysis
2. Gaussian filtering
3. Morphological opening and closing
4. Connected components and contour features
5. MOG2 background subtraction for fixed-camera video
6. Farneback dense optical flow
7. HOG, LBP, and colour-histogram features
8. RBF-SVM fire/smoke/normal classifier
9. False-positive set: sunset, lamps, welding, steam, clouds, reflections
10. Baseline metrics and failure analysis

Exit criterion: classical baseline results and saved thresholds/models.

### Detector Training: Deep-learning baselines and detector — implemented

Implemented:

1. MobileNetV3-Small transfer-learning classifier
2. Optional VGG16 transfer-learning syllabus comparison
3. Lightweight YOLO nano detector for `fire` and `smoke`
4. Mixed-precision training
5. 512 vs 640 input-size experiment
6. Confidence/IoU threshold tuning
7. Grad-CAM for classifier failure analysis
8. Precision, recall, F1, AP50, and AP50-95 evaluation

Exit criterion: best detector checkpoint and reproducible experiment logs. See `docs/PHASE3_IMPLEMENTATION.md`.

### Video Fusion: Temporal video fusion — implemented

Implemented:

1. Detector inference on video/webcam
2. NMS output handling
3. SORT tracking
4. Kalman-filter box smoothing
5. Hungarian IoU association
6. Exponential confidence smoothing
7. N-of-M temporal voting
8. Optical-flow motion evidence
9. Optional MOG2 evidence for stationary cameras
10. Visual risk indicator using persistence, area, motion, and growth

Exit criterion: stable video events with fewer false alerts than frame-only detection. See `docs/PHASE4_IMPLEMENTATION.md`.

### Phase 5: Evaluation, deployment, and case-study report

Implement:

1. Internal test-set evaluation
2. Cross-dataset MIVIA video evaluation
3. Team-recorded phone/webcam evaluation
4. Component ablation study
5. False alarms per minute
6. Event recall and time-to-detection
7. FPS, latency, VRAM, and model-size benchmarking
8. ONNX export and inference comparison
9. Demo interface and event log
10. Final report, charts, limitations, and presentation

Exit criterion: complete case-study evidence, not merely a working demo.

---

# Run Data Prep

## 1. Create the environment

Linux/macOS:

```bash
cd fire-smoke-cv-case-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-data.txt
python scripts/check_environment.py
```

Windows PowerShell:

```powershell
cd fire-smoke-cv-case-study
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-data.txt
python scripts/check_environment.py
```

PyTorch is intentionally not required until Detector Training. Install its CUDA build later using the official selector for your operating system and current driver.

## 2. Download and arrange D-Fire

Use the dataset publisher's pre-split or Kaggle-ready download. Data Prep expects this logical layout by default:

```text
data/raw/dfire/
├── images/
│   ├── image_0001.jpg
│   └── ...
└── labels/
    ├── image_0001.txt
    └── ...
```

Nested folders are supported as long as `images/` and `labels/` mirror one another:

```text
data/raw/dfire/images/train/a.jpg
data/raw/dfire/labels/train/a.txt
```

A missing `.txt` is treated as a negative image for D-Fire because that behaviour is enabled in `configs/data.yaml`.

## 3. Add DFS later

DFS uses Pascal VOC XML. Arrange it as:

```text
data/raw/dfs/
├── images/
└── annotations/
```

Then uncomment the DFS source in `configs/data.yaml`. VOC class names `fire` and `smoke` are converted to IDs 0 and 1.

## 4. Run the preparation pipeline

```bash
python scripts/run_data.py --config configs/data.yaml
```

Generated dataset:

```text
data/processed/fire_smoke/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Generated audit evidence:

```text
reports/data/
├── REPORT.md
├── summary.json
├── manifest.csv
├── rejected.csv
├── duplicates.csv
└── preview.jpg
```

Review `preview.jpg`, `rejected.csv`, and `duplicates.csv` before accepting the dataset.

## 5. Camera calibration

Print or display a checkerboard with **9 × 6 internal corners**. Capture at least 15–20 varied views:

```bash
python scripts/capture_checkerboard.py --camera 0 --cols 9 --rows 6 --count 20
```

Calibrate the camera. Replace `25` with your checkerboard square size in millimetres:

```bash
python scripts/calibrate_camera.py \
  --images data/raw/calibration \
  --cols 9 \
  --rows 6 \
  --square-mm 25
```

Outputs:

```text
artifacts/calibration/camera_calibration.npz
artifacts/calibration/camera_calibration.json
```

Calibration is a syllabus experiment, not a requirement for downloading or cleaning D-Fire.

## Data Prep completion checklist

- [ ] Environment check passes
- [ ] D-Fire is present
- [ ] Pipeline completes without an exception
- [ ] `rejected.csv` is reviewed
- [ ] Annotation preview is manually checked
- [ ] No duplicate group occurs in more than one split
- [ ] Split sizes and class signatures are acceptable
- [ ] `data.yaml` points to the processed dataset
- [ ] At least 10 valid calibration images are found
- [ ] Calibration reprojection error is recorded

## Important scope decisions

- The model is not trained in Data Prep.
- Images are not resized during dataset cleaning; the detector will letterbox them during Detector Training.
- Near-duplicate images are not automatically deleted because perceptual hashing can produce false matches. They are grouped into the same split instead.
- Missing annotations are considered negative only for sources explicitly configured that way.
- Dataset cleaning is deterministic with seed 42.

---

# Run Classical ML

Classical ML compares two non-deep-learning approaches:

1. A data-derived colour, morphology, and contour baseline
2. A three-class RBF-SVM using HOG, LBP, HSV, YCrCb, channel statistics, and edge-contour features

The patch caps in `configs/classical.yaml` keep the RBF-SVM experiment practical on a student laptop.

## 1. Install Classical ML dependencies

With the Data Prep environment activated:

```bash
pip install -r requirements-classical.txt
python scripts/check_classical_environment.py
```

## 2. Confirm Data Prep output exists

Classical ML expects:

```text
data/processed/fire_smoke/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

Do not rebuild random splits in Classical ML. Every generated patch inherits the split of its source image.

## 3. Run the complete Classical ML pipeline

```bash
python scripts/run_classical.py --config configs/classical.yaml
```

The command performs all of the following:

1. Generates fire, smoke, and normal classification patches
2. Fits colour statistics using training patches only
3. Tunes fire/smoke mask-area thresholds using validation patches only
4. Evaluates the classical baseline on the untouched test patches
5. Extracts deterministic HOG, LBP, colour, and contour features
6. Trains RBF-SVM candidates on the training split
7. Selects the SVM using validation macro F1
8. Evaluates the selected SVM on the test split
9. Produces confusion matrices, prediction files, failure galleries, and a comparison table

## 4. Classical ML outputs

```text
data/interim/classical_patches/
├── manifest.csv
├── train/{fire,smoke,normal}/
├── val/{fire,smoke,normal}/
└── test/{fire,smoke,normal}/

artifacts/classical/
├── colour_thresholds.json
├── feature_metadata.json
├── features_train.npz
├── features_val.npz
├── features_test.npz
├── hog_lbp_colour_rbf_svm.joblib
└── svm_training_summary.json

reports/classical/
├── REPORT.md
├── summary.json
├── method_comparison.csv
├── classical_threshold_trials.csv
├── classical_test_metrics.json
├── classical_test_confusion_matrix.png
├── classical_failure_gallery.png
├── svm_validation_trials.csv
├── svm_test_metrics.json
├── svm_test_confusion_matrix.png
└── svm_failure_gallery.png
```

## 5. Run Classical ML as separate steps

Create patches:

```bash
python scripts/prepare_classical_patches.py --config configs/classical.yaml
```

Fit the colour/morphology baseline:

```bash
python scripts/run_classical_baseline.py --config configs/classical.yaml
```

Extract features:

```bash
python scripts/extract_features.py --config configs/classical.yaml --split all
```

Train and evaluate the SVM:

```bash
python scripts/train_svm.py --config configs/classical.yaml
```

## 6. Test the classical video branch

After `colour_thresholds.json` exists:

```bash
python scripts/run_classical_video.py \
  --source path/to/video.mp4 \
  --fixed-camera \
  --output reports/classical/classical_example.mp4 \
  --evidence-csv reports/classical/classical_video_evidence.csv
```

Use `--fixed-camera` only for a stationary webcam or CCTV-style video because it enables MOG2 background subtraction. Omit it for handheld phone footage. Farneback optical flow is calculated in both cases.

Webcam example:

```bash
python scripts/run_classical_video.py --source 0 --fixed-camera
```

## Classical ML completion checklist

- [ ] All three patch classes exist in train, validation, and test
- [ ] Patch manifest is reviewed for visibly incorrect normal crops
- [ ] Colour thresholds were fitted only on training patches
- [ ] Area thresholds were selected only on validation patches
- [ ] SVM hyperparameters were selected only on validation patches
- [ ] Test confusion matrices are generated
- [ ] Failure galleries are manually categorised
- [ ] At least one fixed-camera video is tested with MOG2
- [ ] At least one moving-camera video is tested without MOG2
- [ ] Classical and SVM limitations are written before starting Detector Training

## Classical ML interpretation rule

A strong SVM result does not mean that the final system can localise fire in full video frames. Classical ML is a controlled patch-classification baseline. Detector Training introduces the lightweight classifier comparison and the actual fire/smoke object detector.

---

# Run Detector Training

Detector Training requires the Data Prep detection dataset and Classical ML patch dataset. Install a CUDA-enabled PyTorch build first, then:

```bash
pip install -r requirements-detector.txt
python scripts/check_detector_environment.py
```

Run the lightweight classifier baseline:

```bash
python scripts/train_classifiers.py --config configs/detector.yaml
```

Train the 512 and 640 YOLO11n trials:

```bash
python scripts/train_detector.py --config configs/detector.yaml
```

Tune validation thresholds and evaluate the detector test split once:

```bash
python scripts/tune_detector_thresholds.py --config configs/detector.yaml
```

Complete pipeline:

```bash
python scripts/run_detector.py --config configs/detector.yaml
```

# Run Video Fusion

Install the temporal-video dependencies:

```bash
source .venv/bin/activate
pip install -r requirements-video.txt
python scripts/check_video_environment.py
```

Run on a video:

```bash
python scripts/run_video.py \
  --config configs/video.yaml \
  --source path/to/video.mp4 \
  --output reports/video/annotated_video.mp4
```

Run on a fixed webcam and enable MOG2:

```bash
python scripts/run_video.py --config configs/video.yaml --source 0 --fixed-camera
```

Outputs are written to `reports/video/`. See `docs/PHASE4_IMPLEMENTATION.md` for the full algorithm and test procedure.
