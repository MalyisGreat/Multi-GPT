#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="python3"
VENV_NAME=".venv"
ENABLE_CUDA="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-exe)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --venv-name)
      VENV_NAME="$2"
      shift 2
      ;;
    --enable-cuda)
      ENABLE_CUDA="true"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="$ROOT_DIR/$VENV_NAME"
VENV_PYTHON="$VENV_PATH/bin/python"

if [[ ! -d "$VENV_PATH" ]]; then
  "$PYTHON_EXE" -m venv "$VENV_PATH"
fi

"$VENV_PYTHON" -m pip install --upgrade pip

if [[ "$ENABLE_CUDA" == "true" ]]; then
  "$VENV_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
else
  "$VENV_PYTHON" -m pip install torch torchvision
fi

"$VENV_PYTHON" -m pip install -r requirements.txt

echo "Bootstrap complete."
echo "Use Python from: $VENV_PYTHON"
