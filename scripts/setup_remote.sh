#!/bin/bash
set -eo pipefail

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
mkdir -p data/raw/dfire
if [ ! -d "data/raw/dfire/images" ]; then
    echo "Downloading D-Fire Dataset from direct link..."
    wget -O data/raw/dfire/dfire.zip "https://my.microsoftpersonalcontent.com/personal/c0bd25b6b048b01d/_layouts/15/download.aspx?UniqueId=b60fe0b2-4bc4-4381-bd43-77ec6af19fc4&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiI1YzZmNmEwNi02NTQ4LTRkODMtYmNkZi0wMjUxM2EzZDIzMjciLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvbXkubWljcm9zb2Z0cGVyc29uYWxjb250ZW50LmNvbUA5MTg4MDQwZC02YzY3LTRjNWItYjExMi0zNmEzMDRiNjZkYWQiLCJleHAiOiIxNzg1ODYxNTA2In0.seaLZifUWZp9H3v7YUbU_8kOvLqClokt-y42H3Os0gwmwSCrvC6d9d65twpJC0by1YdC4r5gr48aP0JaHZ6XCSgq5aXX-ZYA6ieELdRGerIkb7d0I0N4l4SpNnPwrR_lIU2H3rS8Qs6qGkqiWfe7fXrT6vYH7nAlDa8Rum1UzB8dxK3kV1D5lt8oboUN4b211qOkzjsdWirqYqWH4if27nhg8K1UGVI2SFpdmRPAaYtdSN_xpj55UKaJRo4hz-TR2jcKIHTlGV72eHmoRBzM32lHFHGWtUsrivzGVNjskKTiM2QYVxf4L2Ngp1WTm-LgPH9lgKfqnqjD_00xTuYVqXMTMeoJS9R-a-wiiHfU0-pdc6ByvmdKdMNTp-s5XNjtSG2Rg5zxGiULq6DTcIPqkKtfYGjngxPkhbBVkET3BtaQ6pX6VMJkkRyhGDWCMJq-L2_VRTO9Lw94JZA6geTat_lNGLJqd0Wb4JpUTtRfvhd-GiNpLEywlUnlyNwtVzjJdkTei1TYtIgGxvU6DC358qGHNqw69icURSD8jUg2Dkk_6LxFObemVtgIDU1RnQzJxXbwt91euQMC7_UKXZq3H70iHMP8G_6xzwiv_CqkOvI.d1uZpMBA5okrv2tPuRRA39cK-8RTV1y0OVhktYInWsU&ApiVersion=2.0"
    echo "Unzipping dataset..."
    unzip -q data/raw/dfire/dfire.zip -d data/raw/dfire/
    rm data/raw/dfire/dfire.zip
else
    echo "Dataset already present, skipping download."
fi

# Create Symlinks so the default config works out of the box
echo "Setting up dataset symlinks..."
mkdir -p data/processed
ln -sfn ../raw/dfire data/processed/fire_smoke

# 7. Install Project Dependencies
echo "Installing FireSmoke project dependencies for Classical ML..."
uv pip install -e ".[classical]"

echo "Setup Complete! You can now run Phase 2 training."
