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
- `src/data/prepare_simple_cifar10_dataset.py`: builds train/val/heldout manifests from CIFAR-10 or CIFAR-100.
- `src/train_native_caption.py`: training script with mixed precision + grad accumulation.
- `src/benchmark_multimodal.py`: benchmark script with BLEU/ROUGE/label-accuracy/swap-sensitivity.
- `src/models/native_vision_gpt2.py`: multimodal model wrapper.

## Quick Start (A40-Ready)

Linux:

```bash
git clone https://github.com/MalyisGreat/Multi-GPT.git
cd Multi-GPT
bash ./scripts/full_run_a40.sh
```

Windows PowerShell:

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

Setup environment (Linux):

```bash
cd Multi-GPT
bash ./scripts/bootstrap.sh --python-exe python3 --venv-name .venv --enable-cuda
```

Setup environment (Windows PowerShell):

```powershell
cd C:/Users/joshj/Multi-GPT
powershell -ExecutionPolicy Bypass -File ./scripts/bootstrap.ps1 -PythonExe "C:/Users/joshj/AppData/Local/Programs/Python/Python312/python.exe" -VenvName ".venv312" -EnableCuda
```

Create dataset (Linux):

```bash
./.venv/bin/python src/data/prepare_simple_cifar10_dataset.py \
  --output-dir data/simple_cifar10_caption \
  --train-size 8000 \
  --val-size 1000 \
  --heldout-size 1000 \
  --overwrite
```

Faster export for large runs:

```bash
./.venv/bin/python src/data/prepare_simple_cifar10_dataset.py \
  --dataset cifar100 \
  --output-dir data/cifar100_full_caption \
  --train-size 45000 \
  --val-size 5000 \
  --heldout-size 10000 \
  --image-format jpg \
  --jpeg-quality 85 \
  --overwrite
```

Create dataset (Windows PowerShell):

```powershell
./.venv312/Scripts/python.exe src/data/prepare_simple_cifar10_dataset.py `
  --output-dir data/simple_cifar10_caption `
  --train-size 8000 `
  --val-size 1000 `
  --heldout-size 1000 `
  --overwrite
```

Train (Linux):

```bash
./.venv/bin/python src/train_native_caption.py \
  --train-jsonl data/simple_cifar10_caption/train.jsonl \
  --val-jsonl data/simple_cifar10_caption/val.jsonl \
  --heldout-jsonl data/simple_cifar10_caption/heldout.jsonl \
  --output-dir checkpoints/full-run \
  --epochs 10 \
  --batch-size 32 \
  --gradient-accumulation-steps 2 \
  --num-workers 8 \
  --precision fp16
```

Train (Windows PowerShell):

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

Benchmark (Linux):

```bash
./.venv/bin/python src/benchmark_multimodal.py \
  --checkpoint-dir checkpoints/full-run \
  --manifest data/simple_cifar10_caption/heldout.jsonl
```

Benchmark (Windows PowerShell):

```powershell
./.venv312/Scripts/python.exe src/benchmark_multimodal.py `
  --checkpoint-dir checkpoints/full-run `
  --manifest data/simple_cifar10_caption/heldout.jsonl
```

## Small Continued Pretrain (Example: Learn "apple")

Create a CIFAR-100 subset dataset that includes `apple`:

```bash
./.venv/bin/python src/data/prepare_simple_cifar10_dataset.py \
  --dataset cifar100 \
  --output-dir data/cifar100_apple_caption \
  --include-labels apple,orange,pear,sweet_pepper \
  --train-size 6000 \
  --val-size 800 \
  --heldout-size 800 \
  --overwrite
```

Continue pretraining from your best checkpoint:

```bash
./.venv/bin/python src/train_native_caption.py \
  --train-jsonl data/cifar100_apple_caption/train.jsonl \
  --val-jsonl data/cifar100_apple_caption/val.jsonl \
  --heldout-jsonl data/cifar100_apple_caption/heldout.jsonl \
  --init-checkpoint checkpoints/full-run/epoch-2.pt \
  --output-dir checkpoints/continued-apple \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation-steps 2 \
  --num-workers 8 \
  --precision fp16 \
  --unfreeze-top-n-blocks 2
```

Benchmark the continued-pretrained checkpoint:

```bash
./.venv/bin/python src/benchmark_multimodal.py \
  --checkpoint-dir checkpoints/continued-apple \
  --checkpoint-path checkpoints/continued-apple/epoch-3.pt \
  --manifest data/cifar100_apple_caption/heldout.jsonl
```

## Shared Token Stream (Unified Mode)

To use a single shared visual+text token stream instead of cross-attention:

```bash
./.venv/bin/python src/train_native_caption.py \
  --train-jsonl data/cifar100_full_caption/train.jsonl \
  --val-jsonl data/cifar100_full_caption/val.jsonl \
  --heldout-jsonl data/cifar100_full_caption/heldout.jsonl \
  --init-checkpoint checkpoints/full-run/epoch-2.pt \
  --output-dir checkpoints/unified-cifar100 \
  --fusion-mode unified \
  --epochs 2 \
  --batch-size 32 \
  --gradient-accumulation-steps 2 \
  --num-workers 8 \
  --precision fp16 \
  --unfreeze-top-n-blocks 2
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

## Image Chat UI

Launch a local chat UI using an exported checkpoint archive:

```powershell
python src/chat_multimodal.py `
  --checkpoint-archive "C:\Users\joshj\Downloads\unified-cifar100-long-best.tar.gz" `
  --extract-root "." `
  --host 127.0.0.1 `
  --port 7860
```

Then open `http://127.0.0.1:7860`, upload an image, and ask a question.
