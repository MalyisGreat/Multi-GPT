# Playbook: Teach GPT-2 Small to Use Images

## Goal

Build a model that takes an image and generates grounded text (first captions, then optional instruction/QA behavior).

## What "Native Image Understanding" Means Here

- GPT-2 receives image token features through cross-attention inside decoder blocks.
- Visual signals are present during each decoding step, not only as a one-time prefix.
- Validation includes image dependence checks (same prompt + different image should change output).

## Strategy Options

### Option A (recommended): Native cross-attention multimodal GPT-2

- Inject cross-attention blocks so GPT-2 attends to visual tokens per layer.
- Train adapters (and optionally top LM layers).
- Pros: better visual grounding and flexibility.
- Cons: higher complexity and compute.

### Option B: Partial/full multimodal finetune

- Unfreeze more of GPT-2 and/or vision encoder.
- Pros: highest potential quality.
- Cons: expensive, unstable without strong data and tuning.

### Option C: Prefix bridge (fallback only)

- Keep CLIP encoder frozen.
- Keep GPT-2 small mostly frozen.
- Train a mapper from image embedding to GPT-2 prefix embeddings.
- Pros: easy baseline.
- Cons: weaker native visual integration.

## Recommended Phased Plan

### Phase 0: Define task and metrics

- Primary task: image captioning.
- Secondary tasks (later): visual question answering, style-conditioned captions.
- Metrics:
  - Automatic: BLEU, CIDEr, ROUGE-L.
  - Human checks: grounding accuracy, hallucination rate.

### Phase 1: Environment + data

- Use the bootstrap script to create `.venv`.
- Warmup dataset (recommended first run): simple CIFAR-10 captions with explicit train/val/heldout split.
- Start with one dataset:
  - MS COCO Captions (good baseline).
  - Optional expansion: Conceptual Captions.
- Standardize preprocessing:
  - Resize/crop images for chosen vision encoder.
  - Clean text, normalize punctuation, set max token length.

### Phase 2: Build native baseline model (Option A)

- Pipeline:
  - `image -> CLIP token sequence -> projector -> GPT-2 cross-attention -> text generation`
- Training recipe:
  - Freeze CLIP.
  - Freeze most GPT-2 weights.
  - Train visual projector + GPT-2 cross-attention parameters with language modeling loss.
- Decoder settings:
  - Start with beam search and compare against nucleus sampling.

### Phase 3: Evaluate and debug

- Quantitative:
  - Track train/val loss.
  - Track BLEU/CIDEr every checkpoint.
- Qualitative:
  - Keep a fixed 50-image probe set.
  - Save generated captions each run to compare regressions.
- Failure analysis:
  - Hallucination: caption mentions objects not in image.
  - Under-description: misses key objects/actions.
  - Generic captions: low specificity.
  - Image-independence: similar output even after image swap.

### Phase 4: Improve

- If baseline works but is weak:
  - Unfreeze top `N` GPT-2 layers.
  - Increase projector capacity.
  - Move to Option B (partial/full multimodal finetune).
  - Use instruction-format data: `<image>\nQuestion: ...\nAnswer: ...`.
- Add data mixtures:
  - Captioning + VQA-style supervised tuning.

### Phase 5: Inference packaging

- Save:
  - Vision encoder config/version.
  - GPT-2 tokenizer/model revision.
  - Projector/cross-attention checkpoint.
  - Generation config JSON.
- Expose a simple inference function:
  - Input: image path (+ optional prompt).
  - Output: generated grounded text.

## Suggested Experiment Matrix

- E1: Cross-attention + frozen GPT-2 backbone + train projector/cross-attn only.
- E2: E1 + unfreeze top 2 GPT-2 blocks.
- E3: E1 + stronger visual projector.
- E4: Partial LM + partial vision unfreeze.
- E5: Full multimodal finetune.

Track all runs with:

- Seed
- Dataset split/hash
- Hyperparameters
- Checkpoint path
- Metrics + qualitative notes

## Minimal Hyperparameter Starting Point

- Batch size: 16 (adjust to VRAM)
- LR: `1e-4` for projector/cross-attention
- Weight decay: `0.01`
- Warmup: `5%`
- Epochs: `5-10`
- Max caption length: `32-64` tokens
- Early stopping: patience `2`

## Acceptance Criteria for v1

- Model produces image-relevant captions on held-out examples.
- Val metrics beat text-only prompt baseline.
- Qualitative probe set shows lower hallucination over training.
- Counterfactual image-swap test changes outputs in expected directions.
- Reproducible run documented with fixed config + checkpoint.
