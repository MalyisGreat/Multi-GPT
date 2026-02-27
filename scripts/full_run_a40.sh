#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="python3"
VENV_NAME=".venv"
DATASET_DIR="data/simple_cifar10_caption"
OUTPUT_DIR="checkpoints/full-run"
TRAIN_SIZE=8000
VAL_SIZE=1000
HELDOUT_SIZE=1000
EPOCHS=10
BATCH_SIZE=32
GRAD_ACCUM=2
NUM_WORKERS=8
PRECISION="fp16"

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
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --train-size)
      TRAIN_SIZE="$2"
      shift 2
      ;;
    --val-size)
      VAL_SIZE="$2"
      shift 2
      ;;
    --heldout-size)
      HELDOUT_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Creating environment + installing deps..."
bash "$ROOT_DIR/scripts/bootstrap.sh" \
  --python-exe "$PYTHON_EXE" \
  --venv-name "$VENV_NAME" \
  --enable-cuda

PYTHON_PATH="$ROOT_DIR/$VENV_NAME/bin/python"
if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python interpreter not found at $PYTHON_PATH"
  exit 1
fi

echo "[2/4] Building dataset (auto-download if needed)..."
"$PYTHON_PATH" src/data/prepare_simple_cifar10_dataset.py \
  --output-dir "$DATASET_DIR" \
  --train-size "$TRAIN_SIZE" \
  --val-size "$VAL_SIZE" \
  --heldout-size "$HELDOUT_SIZE" \
  --overwrite

echo "[3/4] Training model..."
"$PYTHON_PATH" src/train_native_caption.py \
  --train-jsonl "$DATASET_DIR/train.jsonl" \
  --val-jsonl "$DATASET_DIR/val.jsonl" \
  --heldout-jsonl "$DATASET_DIR/heldout.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --precision "$PRECISION"

echo "[4/4] Running multimodal benchmark..."
"$PYTHON_PATH" src/benchmark_multimodal.py \
  --checkpoint-dir "$OUTPUT_DIR" \
  --manifest "$DATASET_DIR/heldout.jsonl"

echo "Full run complete."
echo "Training artifacts: $OUTPUT_DIR"
echo "Benchmark report: $OUTPUT_DIR/benchmark_report.json"
