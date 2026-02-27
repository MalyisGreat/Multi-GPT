from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import evaluate
import sacrebleu
import torch
from PIL import Image
from tqdm import tqdm

from models.native_vision_gpt2 import NativeVisionGPT2, load_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multimodal caption quality.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--manifest", type=str, required=True, help="Heldout/eval JSONL manifest.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--swap-samples", type=int, default=128)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--predictions-jsonl", type=str, default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("epoch-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    def epoch_num(path: Path) -> int:
        m = re.search(r"epoch-(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    candidates = sorted(candidates, key=epoch_num)
    return candidates[-1]


def _load_model_and_processors(
    checkpoint_dir: Path,
    checkpoint_path: Optional[Path],
    device: torch.device,
):
    run_config_path = checkpoint_dir / "run_config.json"
    run_config = {}
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    vision_model_name = run_config.get("vision_model_name", "openai/clip-vit-base-patch32")
    lm_model_name = run_config.get("lm_model_name", "gpt2")
    fusion_mode = run_config.get("fusion_mode", "cross_attn")
    unfreeze_top_n = int(run_config.get("unfreeze_top_n_blocks", 0))

    model = NativeVisionGPT2(
        vision_model_name=vision_model_name,
        lm_model_name=lm_model_name,
        fusion_mode=fusion_mode,
        freeze_vision=not bool(run_config.get("no_freeze_vision", False)),
        freeze_lm_backbone=not bool(run_config.get("no_freeze_lm_backbone", False)),
        unfreeze_top_n_gpt2_blocks=unfreeze_top_n,
    )
    model.to(device)

    if checkpoint_path is None:
        checkpoint_path = _latest_epoch_checkpoint(checkpoint_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"Warning: missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading checkpoint: {len(unexpected)}")

    components = load_processors(
        lm_model_name=lm_model_name,
        vision_model_name=vision_model_name,
    )
    tokenizer = components.tokenizer
    image_processor = components.image_processor

    tokenizer.padding_side = "left"
    model.lm.config.pad_token_id = tokenizer.pad_token_id
    model.lm.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, image_processor, checkpoint_path


def _generate_captions(
    model: NativeVisionGPT2,
    tokenizer,
    image_processor,
    rows: Sequence[Dict[str, str]],
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
    progress_desc: str = "Generate captions",
) -> List[str]:
    model.eval()
    predictions: List[str] = []

    # Avoid using EOS as the initial token when EOS is also the pad token for GPT-2,
    # which can trigger repeated decoder right-padding warnings during generation.
    seed_ids = tokenizer.encode("a", add_special_tokens=False)
    start_token_id = seed_ids[0] if seed_ids else tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None and start_token_id == tokenizer.pad_token_id:
        alt_ids = tokenizer.encode(".", add_special_tokens=False)
        if alt_ids:
            start_token_id = alt_ids[0]
        else:
            start_token_id = tokenizer.eos_token_id

    batch_starts = range(0, len(rows), batch_size)
    for start in tqdm(batch_starts, desc=progress_desc, leave=False):
        batch_rows = rows[start : start + batch_size]
        images = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as img:
                images.append(img.convert("RGB"))

        image_inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)

        input_ids = torch.full(
            (len(batch_rows), 1),
            fill_value=start_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            generated = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_new = generated[:, input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(generated_new, skip_special_tokens=True)
        decoded = [x.strip() for x in decoded]
        predictions.extend(decoded)

    return predictions


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _label_keyword_accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    hits = 0
    for pred, label in zip(predictions, labels):
        if label.lower() in pred.lower():
            hits += 1
    return hits / len(predictions)


def _exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    hits = 0
    for pred, ref in zip(predictions, references):
        if _normalize_text(pred) == _normalize_text(ref):
            hits += 1
    return hits / len(predictions)


def _image_swap_sensitivity(
    model: NativeVisionGPT2,
    tokenizer,
    image_processor,
    rows: Sequence[Dict[str, str]],
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> float:
    if len(rows) < 2:
        return 0.0

    base_rows = list(rows)
    swapped_rows = list(rows)
    rotated_paths = [rows[(i + 1) % len(rows)]["image_path"] for i in range(len(rows))]
    for i in range(len(swapped_rows)):
        swapped_rows[i] = dict(swapped_rows[i])
        swapped_rows[i]["image_path"] = rotated_paths[i]

    base_preds = _generate_captions(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        rows=base_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        device=device,
        progress_desc="Swap sensitivity (base)",
    )
    swap_preds = _generate_captions(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        rows=swapped_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        device=device,
        progress_desc="Swap sensitivity (swapped)",
    )

    changed = 0
    for a, b in zip(base_preds, swap_preds):
        if _normalize_text(a) != _normalize_text(b):
            changed += 1
    return changed / len(base_preds)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, image_processor, resolved_ckpt = _load_model_and_processors(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    rows = load_jsonl(manifest_path)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    predictions = _generate_captions(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        rows=rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
        progress_desc="Benchmark generation",
    )

    references = [r["caption"] for r in rows]
    labels = [r.get("label", "") for r in rows]

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    rouge_metric = evaluate.load("rouge")
    rouge = rouge_metric.compute(predictions=predictions, references=references)
    label_acc = _label_keyword_accuracy(predictions, labels)
    exact = _exact_match(predictions, references)

    swap_rows = rows[: min(args.swap_samples, len(rows))]
    swap_sensitivity = _image_swap_sensitivity(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        rows=swap_rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(resolved_ckpt),
        "manifest": str(manifest_path),
        "n_samples": len(rows),
        "device": str(device),
        "metrics": {
            "bleu": bleu,
            "rougeL": rouge.get("rougeL", 0.0),
            "exact_match": exact,
            "label_keyword_accuracy": label_acc,
            "image_swap_sensitivity": swap_sensitivity,
        },
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else checkpoint_dir / "benchmark_report.json"
    )
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_jsonl = (
        Path(args.predictions_jsonl).resolve()
        if args.predictions_jsonl
        else checkpoint_dir / "benchmark_predictions.jsonl"
    )
    with predictions_jsonl.open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, predictions):
            out = {
                "image_path": row["image_path"],
                "reference": row["caption"],
                "prediction": pred,
                "label": row.get("label", ""),
            }
            f.write(json.dumps(out) + "\n")

    print(json.dumps(report, indent=2))
    print(f"Saved report to: {output_json}")
    print(f"Saved predictions to: {predictions_jsonl}")


if __name__ == "__main__":
    main()
