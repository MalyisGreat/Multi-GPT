# Architecture Plan

## Project

- Name: GPT-2 Small Vision Extension
- Date started: 2026-02-27
- Owner: Josh + Codex
- Status: Draft v2
- Scope: Add image understanding to `gpt2` for image-conditioned text generation (starting with captioning, then optional VQA-style instruction tuning).

## Vision and Context

- Problem statement: `gpt2` is text-only and cannot directly interpret image pixels.
- Native-image target: visual token features must be available to GPT-2 through learned cross-attention layers during generation.
- In-scope:
  - Multimodal architecture that combines an image encoder with GPT-2.
  - Training pipeline for image-text supervision.
  - Evaluation loop for caption quality and grounding.
- Out-of-scope:
  - Training a brand-new LLM from scratch.
  - Production infra hardening.
  - Large-scale multi-node training in v1.
- Primary assumptions:
  - We can use pretrained vision backbones (CLIP/ViT).
  - We can initialize GPT-2 with added cross-attention and train only multimodal parts first.
  - Single-GPU prototyping is acceptable for v1.
- Constraints:
  - GPT-2 small has limited capacity (~124M params).
  - Image grounding quality depends heavily on dataset quality.
  - Compute budget likely favors parameter-efficient adaptation.

## Architecture Overview

- Components and boundaries:
  - Vision encoder: pretrained CLIP ViT (frozen initially).
  - Visual projector: maps CLIP token hidden states -> GPT-2 hidden size.
  - Language model: pretrained GPT-2 small with `add_cross_attention=True`.
  - Data pipeline: image-text pairs with tokenization and image preprocessing.
  - Trainer/evaluator: loss computation, checkpointing, caption metrics.
- Responsibility map:
  - Vision encoder extracts image semantics.
  - Visual projector aligns visual token space to GPT-2 hidden space.
  - GPT-2 autoregressively generates text while attending to visual tokens at each decoder block.
- Data flow:
  - Image -> vision encoder -> visual token sequence -> projector -> GPT-2 cross-attention -> generation.
- External dependencies:
  - PyTorch, Transformers, Datasets, Accelerate.

## Technical Design

- Language/framework stack: Python, PyTorch, Hugging Face Transformers + Datasets, Accelerate.
- Core interfaces:
  - `Dataset.__getitem__ -> {pixel_values, input_ids, attention_mask, labels}`
  - `VisionProjector.forward(vision_tokens) -> encoder_hidden_states`
  - `NativeVisionGPT2.forward(pixel_values, input_ids, labels) -> loss/logits`
- Persistence/storage model:
  - Local datasets cache.
  - Checkpoints at each epoch + best-on-val checkpoint.
  - JSONL experiment logs.
- State and consistency model:
  - Deterministic seeds for reproducibility.
  - Explicit config snapshots saved with each run.
- Deployment topology:
  - Local workstation with optional cloud GPU handoff.
- Non-functional requirements:
  - Availability: N/A for research phase.
  - Latency: target <2s for short caption generation on single image in inference mode.
  - Scale: 10k-1M image-text pairs depending on phase.
  - Cost: start by training projector + cross-attention only, then scale up if needed.

## Decision Log

- Decision ID: D-001
- Date: 2026-02-27
- Owner: Josh + Codex
- Decision: Start with ClipCap-style prefix bridging (frozen CLIP + frozen GPT-2 + trainable mapper).
- Rationale: Lowest complexity, fast validation of multimodal conditioning.
- Alternatives considered: full cross-attention finetune; end-to-end finetune.
- Risks: ceiling on performance for complex reasoning.
- Status: retired

- Decision ID: D-002
- Date: 2026-02-27
- Owner: Josh + Codex
- Decision: Use captioning as first target task before VQA/instruction tuning.
- Rationale: Simple objective, easier debugging and metric tracking.
- Alternatives considered: direct VQA from day 1.
- Risks: may under-represent question-answer behavior.
- Status: active

- Decision ID: D-003
- Date: 2026-02-27
- Owner: Josh + Codex
- Decision: Add staged roadmap (Bridge-only -> Cross-attention adapters -> Optional partial LM unfreeze).
- Rationale: Balances risk and performance gains incrementally.
- Alternatives considered: jump to full finetune.
- Risks: extra engineering steps.
- Status: superseded

- Decision ID: D-004
- Date: 2026-02-27
- Owner: Josh + Codex
- Decision: Make cross-attention native multimodal path the default (CLIP tokens + GPT-2 cross-attention).
- Rationale: Ensures model uses image token context directly during every decoding layer.
- Alternatives considered: prefix-only bridging.
- Risks: added training complexity; randomly initialized cross-attention weights.
- Status: active

- Decision ID: D-005
- Date: 2026-02-27
- Owner: Josh + Codex
- Decision: Define v1 success as grounded captioning with explicit image dependence tests (same prompt, different image).
- Rationale: Prevents false success from text priors.
- Alternatives considered: metric-only validation.
- Risks: requires additional evaluation harness.
- Status: active

- Decision ID: D-006
- Date: 2026-02-27
- Owner: Josh + Codex
- Decision: Use a simple CIFAR-10 caption dataset for warmup training with explicit train/val/heldout splits.
- Rationale: fast setup, low complexity, and immediate heldout evaluation path before moving to richer caption corpora.
- Alternatives considered: jump directly to COCO.
- Risks: synthetic/simple captions limit language richness.
- Status: active

## Update Cadence

- Trigger: changes in architecture, requirements, datasets, or deployment target.
- Review cadence: after each milestone completion.
- Who approves updates: project owner (Josh).
- Last reviewed: 2026-02-27

## Open Risks and Issues

- Risk: Dataset mismatch (caption quality or domain mismatch).
  - Impact: Poor visual grounding.
  - Mitigation: Start with COCO/Conceptual Captions; run qualitative audits.
  - Owner: Josh
  - Due date: 2026-03-06
- Risk: GPT-2 capacity limits.
  - Impact: Weak reasoning or hallucinated captions.
  - Mitigation: Add adapter layers, better decoding constraints, or move to larger base model later.
  - Owner: Josh
  - Due date: 2026-03-20
- Risk: Overfitting projector/cross-attention layers.
  - Impact: Good train loss but weak generalization.
  - Mitigation: holdout validation, data augmentation, early stopping.
  - Owner: Josh
  - Due date: 2026-03-10
- Risk: Weak visual usage despite multimodal architecture.
  - Impact: model ignores image and relies on language priors.
  - Mitigation: counterfactual evaluation (same text prompt, shuffled images), image ablation tests.
  - Owner: Josh
  - Due date: 2026-03-13

## Change History

- 2026-02-27 | Created initial architecture plan | Project kickoff
- 2026-02-27 | Updated to native cross-attention default path | User requested native image understanding
- 2026-02-27 | Added simple CIFAR-10 warmup dataset with heldout split | User requested simple dataset setup before training
