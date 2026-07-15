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
try