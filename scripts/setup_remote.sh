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

# 4. Install Kaggle CLI
echo "Installing Kaggle CLI..."
# Using uv to install globally on the system python
uv pip install --system kaggle

# 5. Download and extract Dataset
echo "Downloading D-Fire Dataset..."
mkdir -p data/raw/dfire
# The -p flag sets the download path
kaggle datasets download -d alxmamaev/dfire-yolo -p data/raw/dfire

echo "Unzipping dataset..."
unzip -q data/raw/dfire/dfire-yolo.zip -d data/raw/dfire/
rm data/raw/dfire/dfire-yolo.zip

# Create Symlinks so the default config works out of the box
echo "Setting up dataset symlinks..."
mkdir -p data/processed
ln -sfn ../../raw/dfire data/processed/fire_smoke

# 6. Install Project Dependencies
echo "Installing FireSmoke project dependencies..."
uv pip install --system -e ".[detector,video]"

echo "Setup Complete! You can now run Phase 2 training."
