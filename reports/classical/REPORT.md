# Classical ML Report: Classical CV and Conventional ML

## Leakage controls & Split notes

- Patches inherit the original Data Prep split (train: 4,500 patches, test: 1,800 patches).
- **Note on validation**: The current dataset uses a direct train/test split (validation patch count is 0). Pixel colour thresholds were fitted on training patches, and classical/ML model evaluations are reported strictly on the 1,800 held-out test patches.
- Final metrics are reported on untouched test patches.

## Patch dataset

| Split | Fire | Smoke | Normal | Total |
|---|---:|---:|---:|---:|
| train | 1500 | 1500 | 1500 | 4500 |
| val | 0 | 0 | 0 | 0 |
| test | 600 | 600 | 600 | 1800 |

## Selected classical thresholds

- Fire mask-area threshold: `0.00500`
- Smoke mask-area threshold: `0.12000`
- Fire pixel vote requirement: `4` of 7 rules
- Smoke pixel vote requirement: `5` of 7 rules

## Results (Held-Out Test Set)

| Method | Split | Accuracy | Macro precision | Macro recall | Macro F1 | Samples |
|---|---|---:|---:|---:|---:|---:|
| colour+morphology | test | 0.3289 | 0.1119 | 0.3289 | 0.1669 | 1800 |
| HOG+LBP+colour RBF-SVM | test | 0.8878 | 0.8882 | 0.8878 | 0.8879 | 1800 |
| LBP+GLCM+Contours Random Forest | test | 0.8667 | 0.8663 | 0.8667 | 0.8665 | 1800 |
| Extra Trees | test | 0.8544 | 0.8547 | 0.8544 | 0.8545 | 1800 |
| XGBoost | test | 0.8789 | 0.8790 | 0.8789 | 0.8789 | 1800 |
| LightGBM | test | 0.8828 | 0.8833 | 0.8828 | 0.8830 | 1800 |

## Visualizations and Charts

- `f1_comparison_bar.png`: Test macro F1 score bar chart ranking all methods.
- `detailed_metrics_bar.png`: Test Precision, Recall, and F1 comparison.
- `radar_chart.png`: Multi-metric polar radar chart comparing performance.
- `roc_curves.png`: Macro ROC curves on test patches.
- `*_test_confusion_matrix.png`: Per-model test confusion matrices.
- `*_failure_gallery.png`: Visual failure galleries highlighting ambiguous and edge cases.

## Interpretation requirement

Do not present the classical system as the final detector. Document failures on sunsets, lamps, clouds, steam, reflections, and low-contrast smoke. Detector Training must test whether learned models improve these weaknesses.
