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

# Detectron2 must be built against the PyTorch already provided by Colab.
echo "Installing Detectron2 without build isolation ..."
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git"

# Install base dependencies first. PHALP, Chumpy and 4D-Humans are excluded
# from requirements.txt so their legacy Git dependencies cannot trigger
# isolated rebuilds of already installed packages.
echo "Installing base requirements ..."
python -m pip install -r requirements.txt

# Chumpy's legacy build metadata is not compatible with modern isolated pip
# builds. Build it in the current environment instead.
echo "Installing Chumpy without build isolation ..."
python -m pip install --no-build-isolation \
  "git+https://github.com/mattloper/chumpy"

# Install 4D-Humans itself without resolving Chumpy again.
echo "Installing 4D-Humans without dependency resolution ..."
python -m pip install --no-deps \
  "git+https://github.com/miu200521358/4D-Humans.git"

# Detectron2 and base dependencies are already installed. Install PHALP itself
# without resolving Detectron2 again.
echo "Installing PHALP without dependency resolution ..."
python -m pip install --no-deps \
  "git+https://github.com/miu200521358/PHALP.git"

echo "Verifying core imports ..."
python - <<'PY'
import torch
import detectron2
import phalp
import hmr2
print("PyTorch OK:", torch.__version__)
print("Detectron2 OK")
print("PHALP OK")
print("HMR2 OK")
PY

echo "============================================================"
echo "Base Colab installation completed."
echo "============================================================"
