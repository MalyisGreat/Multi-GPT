# Multi-GPT (GPT-2 + Vision)

Native multimodal GPT-2 training pipeline with:

- CLIP vision encoder
- GPT-2 with cross-attention (`add_cross_attention=True`)
- Auto dataset build/download
- Heldout split support
- Multimodal benchmark reporting

## Project Files

- `scripts/bootstrap.ps1`: environment setup + dependency install (CUDA-capable option).
- `scripts/full_run_a40.ps1`: one-command setup/download/train/benchmark pipeline.
- `src/data/prepare_simple_cifar10_dataset.py`: builds train/val/heldout dataset manifests.
- `src/train_native_caption.py`: training script with mixed precision + grad accumulation.
- `src/benchmark_multimodal.py`: benchmark script with BLEU/ROUGE/label-accuracy/swap-sensitivity.
- `src/models/native_vision_gpt2.py`: multimodal model wrapper.

## Quick Start (A40-Ready)

Run everything end-to-end:

```powershell
cd C:/Users/joshj/Multi-GPT
powershell -ExecutionPolicy Bypass -File ./scripts/full_run_a40.ps1
```

This will:

1. Create `.venv312` and install CUDA PyTorch.
2. Auto-download and build the CIFAR-10 caption dataset.
3. Train the multimodal model.
4. Run a benchmark on heldout images.

## Manual Step-by-Step

Setup environment:

```powershell
cd C:/Users/joshj/Multi-GPT
powershell -ExecutionPolicy Bypass -File ./scripts/bootstrap.ps1 -PythonExe "C:/Users/joshj/AppData/Local/Programs/Python/Python312/python.exe" -VenvName ".venv312" -EnableCuda
```

Create dataset:

```powershell
./.venv312/Scripts/python.exe src/data/prepare_simple_cifar10_dataset.py `
  --output-dir data/simple_cifar10_caption `
  --train-size 8000 `
  --val-size 1000 `
  --heldout-size 1000 `
  --overwrite
```

Train:

```powershell
./.venv312/Scripts/python.exe src/train_native_caption.py `
  --train-jsonl data/simple_cifar10_caption/train.jsonl `
  --val-jsonl data/simple_cifar10_caption/val.jsonl `
  --heldout-jsonl data/simple_cifar10_caption/heldout.jsonl `
  --output-dir checkpoints/full-run `
  --epochs 10 `
  --batch-size 32 `
  --gradient-accumulation-steps 2 `
  --num-workers 8 `
  --precision fp16
```

Benchmark:

```powershell
./.venv312/Scripts/python.exe src/benchmark_multimodal.py `
  --checkpoint-dir checkpoints/full-run `
  --manifest data/simple_cifar10_caption/heldout.jsonl
```

## Benchmark Output

Generated files in checkpoint directory:

- `benchmark_report.json`
- `benchmark_predictions.jsonl`

Core metrics:

- `bleu`
- `rougeL`
- `exact_match`
- `label_keyword_accuracy`
- `image_swap_sensitivity`
