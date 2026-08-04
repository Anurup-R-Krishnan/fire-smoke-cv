# Lightning Studio Classical ML Setup

## Goal
Automate the setup process for the Classical ML pipeline on Lightning Studio. The user only wants to run classical ML model training and comparison.

## Proposed Changes

### 1. Update `scripts/setup_remote.sh`
- Modify the dependency installation step.
- Currently, it installs `.[detector,video]`.
- Change this to `.[classical]` to ensure `scikit-learn`, `scikit-image` (for GLCM), and other classical ML dependencies are installed.

### 2. Create Lightning Studio Hook
- Create a new script: `.lightningstudio/on_start.sh`.
- This is the standard initialization hook for Lightning AI Studio instances.
- The script will simply execute `bash scripts/setup_remote.sh` so that dependencies and Kaggle datasets are automatically handled upon starting the Studio.

## Verification
- Review the modified `setup_remote.sh` to ensure `detector` and `video` dependencies are removed and replaced with `classical`.
- Ensure `.lightningstudio/on_start.sh` has the correct executable permissions and correct path.
