#!/bin/bash
set -e

echo "🚀 Bootstrapping FireSmoke CV Project for Remote GPU Environment..."

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing 'uv' package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Install dependencies (assuming we are already in the repo directory)
echo "🐍 Installing Python dependencies..."
uv pip install --system -e ".[detector]"

# 3. Create required data directory structures
echo "📁 Setting up data directories..."
mkdir -p data/raw data/interim data/processed/fire_smoke

echo "✅ Setup complete! You are ready to start training."
echo ""
echo "🔥 NEXT STEPS:"
echo "1. Upload your dataset or download it via Kaggle API to a folder (e.g., /teamspace/studios/this_studio/dfire-yolo)"
echo "2. Symlink the dataset: ln -s /path/to/downloaded/dataset data/processed/fire_smoke"
echo "3. Run training: python scripts/run_detector_pipeline.py --config configs/default.yaml"
