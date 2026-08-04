#!/bin/bash
set -eo pipefail

# 1. Error Handling & Prerequisites
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "ERROR: Missing Kaggle credentials."
    echo "Please set them before running this script:"
    echo "  export KAGGLE_USERNAME=\"your_username\""
    echo "  export KAGGLE_KEY=\"your_api_key\""
    exit 1
fi

echo "Starting Remote Setup for Lightning AI..."

# 2. System Dependencies
echo "Installing system dependencies for OpenCV/YOLO..."
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1 libglib2.0-0 unzip

# 3. Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv is already installed."
fi

# 4. Create and activate virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv .venv
fi
source .venv/bin/activate

# 5. Install Kaggle CLI
echo "Installing Kaggle CLI..."
uv pip install kaggle

# 6. Download and extract Dataset
echo "Downloading D-Fire Dataset..."
mkdir -p data/raw/dfire
kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo -p data/raw/dfire --unzip

# Create Symlinks so the default config works out of the box
echo "Setting up dataset symlinks..."
mkdir -p data/processed
ln -sfn ../../raw/dfire data/processed/fire_smoke

# 7. Install Project Dependencies
echo "Installing FireSmoke project dependencies for Classical ML..."
uv pip install -e ".[classical]"

echo "Setup Complete! You can now run Phase 2 training."
