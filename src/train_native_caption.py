from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.native_vision_gpt2 import NativeVisionGPT2, load_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train native-vision GPT-2 caption model.")
    parser.add_argument("--train-jsonl", type=str, required=True, help="JSONL with image_path + caption.")
    parser.add_argument("--val-jsonl", type=str, default="", help="Optional validation JSONL.")
    parser.add_argument("--heldout-jsonl", type=str, default="", help="Optional heldout JSONL.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/native-vision-gpt2")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to initialize model weights for continued pretraining.",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-tf32", action="store_true")

    parser.add_argument("--vision-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--lm-model-name", type=str, default="gpt2")
    parser.add_argument("--unfreeze-top-n-blocks", type=int, default=0)
    parser.add_argument("--no-freeze-vision", action="store_true")
    parser.add_argument("--no-freeze-lm-backbone", action="store_true")

    parser.add_argument(
        "--eval-heldout-each-epoch",
        action="store_true",
        help="If set, computes heldout loss every epoch in addition to final heldout evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass
class CaptionExample:
    image_path: str
    caption: str


class CaptionJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        rows = load_jsonl(jsonl_path)
        self.examples = [
            CaptionExample(image_path=row["image_path"], caption=row["caption"]) for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> CaptionExample:
        return self.examples[idx]


class CaptionCollator:
    """
    Top-level callable collator so DataLoader workers can pickle it on Windows.
    """

    def __init__(self, tokenizer, image_processor, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __call__(self, batch: List[CaptionExample]) -> Dict[str, torch.Tensor]:
        captions = [x.caption for x in batch]
        images = []
        for x in batch:
            with Image.open(x.image_path) as img:
                images.append(img.convert("RGB"))

        image_inputs = self.image_processor(images=images, return_tensors="pt")
        text_inputs = self.tokenizer(
            captions,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = text_inputs["input_ids"].clone()
        labels[text_inputs["attention_mask"] == 0] = -100

        return {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
        }


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


@torch.no_grad()
def evaluate(
    model: NativeVisionGPT2,
    dataloader: DataLoader,
    device: torch.device,
    precision: str,
    desc: str = "Eval",
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast_context(device, precision):
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        total_loss += outputs.loss.item()
        total_steps += 1

    if total_steps == 0:
        return 0.0
    return total_loss / total_steps


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    components = load_processors(
        lm_model_name=args.lm_model_name,
        vision_model_name=args.vision_model_name,
    )
    tokenizer = components.tokenizer
    image_processor = components.image_processor

    model = NativeVisionGPT2(
        vision_model_name=args.vision_model_name,
        lm_model_name=args.lm_model_name,
        freeze_vision=not args.no_freeze_vision,
        freeze_lm_backbone=not args.no_freeze_lm_backbone,
        unfreeze_top_n_gpt2_blocks=args.unfreeze_top_n_blocks,
    )
    model.to(device)

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).resolve()
        checkpoint = torch.load(init_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            init_state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            init_state = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {init_path}")
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(init_path),
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                }
            )
        )

    train_dataset = CaptionJsonlDataset(args.train_jsonl)
    val_dataset: Optional[CaptionJsonlDataset] = None
    if args.val_jsonl:
        val_dataset = CaptionJsonlDataset(args.val_jsonl)
    heldout_dataset: Optional[CaptionJsonlDataset] = None
    if args.heldout_jsonl:
        heldout_dataset = CaptionJsonlDataset(args.heldout_jsonl)

    collate_fn = CaptionCollator(tokenizer, image_processor, args.max_length)
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn,
        )

    heldout_loader = None
    if heldout_dataset is not None:
        heldout_loader = DataLoader(
            heldout_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    use_scaler = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["device"] = str(device)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with _autocast_context(device, args.precision):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss_to_backprop = loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            should_step = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_params, args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            progress.set_postfix(train_loss=running_loss / step)

        epoch_train_loss = running_loss / max(1, len(train_loader))
        metrics = {"epoch": epoch, "train_loss": epoch_train_loss}

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, args.precision, desc="Validation")
            metrics["val_loss"] = val_loss
        if heldout_loader is not None and args.eval_heldout_each_epoch:
            heldout_loss = evaluate(model, heldout_loader, device, args.precision, desc="Heldout")
            metrics["heldout_loss"] = heldout_loss

        print(json.dumps(metrics))

        ckpt_path = output_dir / f"epoch-{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "run_config": run_config,
            },
            ckpt_path,
        )

    model.lm.config.pad_token_id = tokenizer.pad_token_id
    model.lm.save_pretrained(output_dir / "gpt2_with_cross_attn")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    torch.save(
        {"visual_projector": model.visual_projector.state_dict()},
        output_dir / "visual_projector.pt",
    )

    if heldout_loader is not None:
        final_heldout_loss = evaluate(model, heldout_loader, device, args.precision, desc="Final heldout")
        print(json.dumps({"final_heldout_loss": final_heldout_loss}))

    print(f"Training complete. Artifacts saved to {output_dir}.")


if __name__ == "__main__":
    main()
