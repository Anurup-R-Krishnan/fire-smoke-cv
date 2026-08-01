# Detector Training Implementation: Deep-Learning Classifiers and Detector

Detector Training assumes that Data Prep has produced the two-class YOLO dataset and Classical ML has produced the three-class patch dataset.

## Inputs

```text
data/processed/fire_smoke/data.yaml
data/processed/fire_smoke/images/{train,val,test}/
data/processed/fire_smoke/labels/{train,val,test}/
data/interim/classical_patches/{train,val,test}/{fire,smoke,normal}/
```

## Experiments

### A. MobileNetV3-Small classifier

Purpose: lightweight transfer-learning baseline for whole-patch classification.

Training is divided into two stages:

1. Freeze the convolutional backbone and train the classifier head.
2. Reload the best validation checkpoint, unfreeze the final feature blocks, and fine-tune at a smaller learning rate.

The selected checkpoint maximizes validation macro F1. Test metrics are reported only after selection.

### B. VGG16 syllabus comparison

The VGG16 pipeline is fully implemented but disabled in `configs/detector.yaml`. Enable it only when the MobileNet and detector experiments are stable. Its default batch size is 8 because a mobile RTX 3050 Ti commonly has limited VRAM.

### C. YOLO11n detector

Two independent detector trials are trained from the same pretrained nano checkpoint:

```text
512 × 512
640 × 640
```

The selected input size maximizes validation mAP50-95. Confidence and NMS IoU are then tuned on the validation set. The test split is evaluated once using the frozen size and thresholds.

## Install

Create or activate the project environment, then install a CUDA-enabled PyTorch build that matches the NVIDIA driver. After that:

```bash
pip install -r requirements-detector.txt
python scripts/check_detector_environment.py
```

Do not begin YOLO training if the checker reports `CUDA available: False`.

## Recommended order

Run the lightweight classifier first:

```bash
python scripts/train_classifiers.py --config configs/detector.yaml
```

Train the two detector sizes:

```bash
python scripts/train_detector.py --config configs/detector.yaml
```

Tune thresholds and evaluate the detector test split:

```bash
python scripts/tune_detector_thresholds.py --config configs/detector.yaml
```

Or run everything:

```bash
python scripts/run_detector.py --config configs/detector.yaml
```

## RTX 3050 Ti controls

The defaults are intentionally conservative:

```text
YOLO batch size: 8
Data-loader workers: 2
Dataset caching: disabled
Automatic mixed precision: enabled
MobileNet batch size: 32
VGG16 batch size: 8
```

When CUDA runs out of memory, reduce the relevant batch size before reducing image resolution:

```yaml
detector:
  batch_size: 4
```

Do not enable dataset RAM caching on a laptop unless enough memory is available.

## Outputs

### Classifier artifacts

```text
artifacts/detector/mobilenet_v3_small_best.pt
artifacts/detector/vgg16_best.pt                 # only if enabled
artifacts/detector/classifier_selection.json
```

### Detector artifacts

```text
artifacts/detector/yolo_runs/yolo11n_512/
artifacts/detector/yolo_runs/yolo11n_640/
artifacts/detector/best_fire_smoke_detector.pt
artifacts/detector/detector_training_summary.json
artifacts/detector/detector_thresholds.json
```

### Reports

```text
reports/detector/classifier_comparison.csv
reports/detector/mobilenet_v3_small/test_metrics.json
reports/detector/mobilenet_v3_small/test_confusion_matrix.png
reports/detector/mobilenet_v3_small/training_curves.png
reports/detector/mobilenet_v3_small/gradcam_failures/
reports/detector/detector_size_comparison.csv
reports/detector/detector_threshold_trials.csv
reports/detector/REPORT.md
```

## Metrics

Classifier:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision, recall and F1
- Macro one-vs-rest ROC-AUC when all classes are represented
- Confusion matrix

Detector:

- Precision
- Recall
- F1 calculated from validation precision and recall
- mAP50
- mAP50-95

## Grad-CAM

Grad-CAM is generated from misclassified test patches. Use it to discuss whether the classifier focused on flames, smoke texture, bright backgrounds, or irrelevant regions. It is an interpretation aid, not proof that the model reasons like a person.

## Methodological rules

1. Never select an architecture or threshold using the test set.
2. Do not report the detector's training-set metrics as final performance.
3. Keep the Data Prep split unchanged.
4. Save false positives, especially sunsets, lamps, steam, clouds, reflections and display screens.
5. Record GPU model, VRAM, package versions, image size and batch size with every result.
6. Video Fusion must use `best_fire_smoke_detector.pt` and `detector_thresholds.json`; it must not retune them on demonstration videos.
