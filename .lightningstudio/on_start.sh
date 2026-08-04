#!/bin/bash
set -e

echo "Running Lightning Studio custom initialization hook..."
# Execute the remote setup script for classical ML pipeline dependencies
bash scripts/setup_remote.sh
