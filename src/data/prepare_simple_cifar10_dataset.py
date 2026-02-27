from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

from torchvision.datasets import CIFAR10


TRAIN_TEMPLATES = [
    "a photo of a {label}.",
    "an image of a {label}.",
    "a picture of a {label}.",
    "a small {label} in a photo.",
]

EVAL_TEMPLATE = "a photo of a {label}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple image-caption dataset from CIFAR-10 with train/val/heldout splits."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/simple_cifar10_caption",
        help="Directory where images and JSONL manifests are written.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/raw",
        help="Directory used by torchvision to download/store CIFAR-10.",
    )
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output-dir before writing.",
    )
    return parser.parse_args()


def _sample_indices(population_size: int, sample_size: int, rng: random.Random) -> List[int]:
    if sample_size > population_size:
        raise ValueError(f"sample_size={sample_size} > population_size={population_size}")
    indices = list(range(population_size))
    rng.shuffle(indices)
    return indices[:sample_size]


def _caption_for_label(label_name: str, idx: int, split: str) -> str:
    if split == "train":
        template = TRAIN_TEMPLATES[idx % len(TRAIN_TEMPLATES)]
        return template.format(label=label_name)
    return EVAL_TEMPLATE.format(label=label_name)


def _write_split(
    split: str,
    dataset: CIFAR10,
    indices: Sequence[int],
    root: Path,
) -> Path:
    image_dir = root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / f"{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row_num, idx in enumerate(indices):
            image, label_id = dataset[idx]
            label_name = dataset.classes[label_id]
            caption = _caption_for_label(label_name, idx=row_num, split=split)

            image_path = image_dir / f"{split}_{idx:05d}.png"
            image.save(image_path)

            record = {
                "image_path": str(image_path.resolve()),
                "caption": caption,
                "label": label_name,
                "source_index": idx,
            }
            manifest_file.write(json.dumps(record) + "\n")

    return manifest_path


def _summarize(
    output_dir: Path,
    class_names: Iterable[str],
    train_size: int,
    val_size: int,
    heldout_size: int,
    seed: int,
) -> None:
    summary = {
        "name": "simple_cifar10_caption",
        "seed": seed,
        "splits": {"train": train_size, "val": val_size, "heldout": heldout_size},
        "class_names": list(class_names),
        "notes": [
            "Train/val come from CIFAR-10 train set.",
            "Heldout comes from CIFAR-10 test set and should not be used for training.",
            "Captions are intentionally simple class descriptions.",
        ],
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir).resolve()
    download_dir = Path(args.download_dir).resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to rebuild the dataset."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CIFAR10(root=str(download_dir), train=True, download=True)
    test_dataset = CIFAR10(root=str(download_dir), train=False, download=True)

    train_total_needed = args.train_size + args.val_size
    train_pool = _sample_indices(len(train_dataset), train_total_needed, rng)
    train_indices = train_pool[: args.train_size]
    val_indices = train_pool[args.train_size :]
    heldout_indices = _sample_indices(len(test_dataset), args.heldout_size, rng)

    train_manifest = _write_split("train", train_dataset, train_indices, output_dir)
    val_manifest = _write_split("val", train_dataset, val_indices, output_dir)
    heldout_manifest = _write_split("heldout", test_dataset, heldout_indices, output_dir)

    _summarize(
        output_dir=output_dir,
        class_names=train_dataset.classes,
        train_size=len(train_indices),
        val_size=len(val_indices),
        heldout_size=len(heldout_indices),
        seed=args.seed,
    )

    print("Dataset setup complete.")
    print(f"train manifest:   {train_manifest}")
    print(f"val manifest:     {val_manifest}")
    print(f"heldout manifest: {heldout_manifest}")


if __name__ == "__main__":
    main()
