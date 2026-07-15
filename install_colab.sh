#!/usr/bin/env bash
set -euo pipefail

# MMDAutoTrace4 Sh-hei Edition
# Minimal installer for Google Colab 2026.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "MMDAutoTrace4 Sh-hei Edition - Colab setup"
echo "============================================================"

python --version

python - <<'PY'
import sys
print("Python executable:", sys.executable)
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch check failed:", e)
PY

# PyTorch 2.11 requires setuptools < 82.
python -m pip install --upgrade pip wheel "setuptools<82"

python - <<'PY'
import setuptools
print("Setuptools:", setuptools.__version__)
PY

if [[ ! -f requirements.txt ]]; then
    echo "ERROR: requirements.txt not found: $PROJECT_DIR/requirements.txt"
    exit 1
fi

# Install Detectron2 first without pip build isolation so the build can see
# the PyTorch already provided by the Colab runtime.
echo "Installing Detectron2 without build isolation ..."
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git"

echo "Installing requirements.txt ..."
python -m pip install -r requirements.txt

echo "============================================================"
echo "Base Colab installation completed."
echo "============================================================"
