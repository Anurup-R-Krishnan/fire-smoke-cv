# Classical ML Implementation: Classical CV and Conventional Machine Learning

## Goal

Classical ML establishes credible non-deep-learning baselines before training the Detector Training detector. It answers four questions:

1. How far can explicit colour and morphology rules go?
2. Do texture and gradient descriptors improve over colour rules?
3. Which failure modes remain unsolved?
4. How much improvement must the deep-learning system demonstrate?

## Data representation

Data Prep produces a two-class object-detection dataset. Classical ML converts its boxes into a three-class patch-classification dataset:

```text
fire
smoke
normal
```

Positive patches are created from labelled boxes with 12% spatial context. Normal patches are sampled from negative images and non-overlapping regions of positive images. Patches inherit the original image split, preventing a source image from contributing to multiple splits.

Default caps:

| Split | Maximum per class |
|---|---:|
| Train | 1,500 |
| Validation | 400 |
| Test | 600 |

These caps are deliberate. RBF-SVM fitting becomes expensive as the sample count grows, and Classical ML is a baseline rather than the final system.

## Baseline A: data-derived colour rules

### Training-only threshold fitting

Pixels are sampled from training fire patches and training smoke patches. The implementation estimates robust quantiles and derives thresholds for:

### Fire evidence

- Red-minus-green intensity
- Green-minus-blue intensity
- HSV saturation
- HSV value
- YCrCb Cr
- YCrCb Cb
- Warm HSV hue range

A pixel is accepted as a fire candidate when at least four of seven rules agree.

### Smoke evidence

- Low-to-moderate HSV saturation
- Valid brightness range
- Small Cr deviation from neutral chroma
- Small Cb deviation from neutral chroma
- Small red-green difference
- Small green-blue difference

A pixel is accepted as a smoke candidate when at least five of seven rules agree.

The threshold model is saved at:

```text
artifacts/classical/colour_thresholds.json
```

### Filtering and region analysis

Each candidate mask undergoes:

1. 3×3 Gaussian smoothing before thresholding
2. Morphological opening
3. Morphological closing
4. External contour extraction
5. Relative-area filtering

Region statistics include:

- Foreground area ratio
- Largest component area
- Perimeter
- Aspect ratio
- Solidity
- Extent
- Component count
- Normalised centroid

### Validation-only decision threshold tuning

Pixel rules create masks, but a patch still requires a decision threshold. Fire and smoke mask-area thresholds are selected using validation macro F1. The test patches are not inspected during this selection.

## Motion branch

### MOG2

MOG2 estimates foreground regions for stationary-camera footage. It is supporting evidence, not an independent fire detector. It should be disabled for handheld footage because camera motion invalidates the stationary-background assumption.

### Farneback dense optical flow

Dense optical flow is calculated between adjacent grayscale frames. The implementation reports:

- Mean flow magnitude
- Magnitude variance
- Moving-pixel ratio
- Dominant direction
- Upward-motion ratio

The Classical ML video script visualises these values. Video Fusion will combine them with learned detections and temporal event logic.

## Baseline B: handcrafted features and RBF-SVM

Each 96×96 patch is represented by:

### HOG

- 9 orientations
- 8×8 pixels per cell
- 2×2 cells per block
- L2-Hys normalisation

### Uniform LBP

- 8 neighbouring samples
- Radius 1
- Normalised texture histogram

### Colour descriptors

- HSV histograms
- YCrCb histograms
- Per-channel mean and standard deviation

### Contour descriptors

Canny edges are summarised using area, perimeter, solidity, extent, aspect ratio, component count, and centroid statistics.

### Classifier

```text
StandardScaler
    ↓
RBF-SVC
```

The SVM uses balanced class weights. Candidate `C` values are evaluated on the validation split, and the model with the highest validation macro F1 is saved.

## Evaluation

Both baselines report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision, recall, and F1
- Three-class confusion matrix
- Prediction CSV
- Failure gallery

Macro F1 is the primary selection metric because the three classes must be treated equally even when the available patch counts differ.

## Required manual failure categories

After running the pipeline, inspect the failure galleries and categorise errors as:

- Sunset or warm sky
- Lamp or headlight
- Welding
- Reflection
- Red/orange object
- Cloud or fog
- Steam
- Very faint smoke
- Very small flame
- Background contamination inside a labelled box

These categories become the error-analysis table in the case-study report.

## Commands

Complete run:

```bash
python scripts/run_classical.py --config configs/classical.yaml
```

Video branch:

```bash
python scripts/run_classical_video.py --source input.mp4 --fixed-camera
```

Tests:

```bash
PYTHONPATH=src pytest -q tests/test_classical_pipeline.py
```

## Classical ML exit criteria

Classical ML is complete only when:

1. The patch manifest contains no obvious split or crop errors.
2. Classical and SVM metrics are generated on the untouched test split.
3. At least one confusion matrix and one failure gallery are discussed.
4. MOG2 and optical flow are demonstrated on video.
5. The report clearly explains why patch classification is not equivalent to object detection.
