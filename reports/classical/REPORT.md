# Classical ML Report: Classical CV and Conventional ML

## Leakage controls

- Patches inherit the original Data Prep train/validation/test split.
- Pixel colour thresholds are fitted using training patches only.
- Mask-area thresholds are selected using validation patches only.
- Final metrics are reported on untouched test patches.

## Patch dataset

| Split | Fire | Smoke | Normal |
|---|---:|---:|---:|
| train | 1500 | 1500 | 1500 |
| val | 0 | 0 | 0 |
| test | 600 | 600 | 600 |

## Selected classical thresholds

- Fire mask-area threshold: `0.00500`
- Smoke mask-area threshold: `0.12000`
- Fire pixel vote requirement: `4` of 7 rules
- Smoke pixel vote requirement: `5` of 7 rules

## Results

| Method | Split | Accuracy | Macro precision | Macro recall | Macro F1 | Samples |
|---|---|---:|---:|---:|---:|---:|
| colour+morphology | val | 0.3289 | 0.1119 | 0.3289 | 0.1669 | 1800 |
| colour+morphology | test | 0.3289 | 0.1119 | 0.3289 | 0.1669 | 1800 |
| HOG+LBP+colour RBF-SVM | val | 0.8878 | 0.8882 | 0.8878 | 0.8879 | 1800 |
| HOG+LBP+colour RBF-SVM | test | 0.8878 | 0.8882 | 0.8878 | 0.8879 | 1800 |
| LBP+GLCM+Contours Random Forest | val | 0.8667 | 0.8663 | 0.8667 | 0.8665 | 1800 |
| LBP+GLCM+Contours Random Forest | test | 0.8667 | 0.8663 | 0.8667 | 0.8665 | 1800 |
| Extra Trees | val | 0.8544 | 0.8547 | 0.8544 | 0.8545 | 1800 |
| Extra Trees | test | 0.8544 | 0.8547 | 0.8544 | 0.8545 | 1800 |
| XGBoost | val | 0.8789 | 0.8790 | 0.8789 | 0.8789 | 1800 |
| XGBoost | test | 0.8789 | 0.8790 | 0.8789 | 0.8789 | 1800 |
| LightGBM | val | 0.8828 | 0.8833 | 0.8828 | 0.8830 | 1800 |
| LightGBM | test | 0.8828 | 0.8833 | 0.8828 | 0.8830 | 1800 |

## Artifacts

- `colour_thresholds.json`: fitted pixel and area thresholds
- `*_model.joblib`: trained ML pipelines
- `features_{train,val,test}.npz`: deterministic feature archives
- Confusion matrices and failure galleries in `reports/classical/`

## Interpretation requirement

Do not present the classical system as the final detector. Document failures on sunsets, lamps, clouds, steam, reflections, and low-contrast smoke. Detector Training must test whether learned models improve these weaknesses.
