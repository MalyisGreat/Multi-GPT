from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100


TRAIN_TEMPLATES = [
    "a photo of a {label}.",
    "an image of a {label}.",
    "a picture of a {label}.",
    "a small {label} in a photo.",
]

EVAL_TEMPLATE = "a photo of a {label}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple image-caption dataset from CIFAR-10/100 with train/val/heldout splits."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar10",
        help="Source dataset to build captions from.",
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
        help="Directory used by torchvision to download/store CIFAR datasets.",
    )
    parser.add_argument(
        "--include-labels",
        type=str,
        default="",
        help="Optional comma-separated class names to keep (for example: apple,orange,pear).",
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


def _sample_indices(candidates: Sequence[int], sample_size: int, rng: random.Random) -> List[int]:
    if sample_size > len(candidates):
        raise ValueError(f"sample_size={sample_size} > available_candidates={len(candidates)}")
    indices = list(candidates)
    rng.shuffle(indices)
    return indices[:sample_size]


def _caption_for_label(label_name: str, idx: int, split: str) -> str:
    if split == "train":
        template = TRAIN_TEMPLATES[idx % len(TRAIN_TEMPLATES)]
        return template.format(label=label_name)
    return EVAL_TEMPLATE.format(label=label_name)


def _write_split(
    split: str,
    dataset: CIFAR10 | CIFAR100,
    indices: Sequence[int],
    root: Path,
) -> Path:
    image_dir = root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / f"{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        iterator = tqdm(indices, total=len(indices), desc=f"Writing {split}", unit="img")
        for row_num, idx in enumerate(iterator):
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
    dataset_name: str,
    class_names: Iterable[str],
    train_size: int,
    val_size: int,
    heldout_size: int,
    seed: int,
    included_labels: Optional[Sequence[str]],
) -> None:
    summary = {
        "name": f"simple_{dataset_name}_caption",
        "dataset": dataset_name,
        "seed": seed,
        "splits": {"train": train_size, "val": val_size, "heldout": heldout_size},
        "class_names": list(class_names),
        "included_labels": list(included_labels) if included_labels else [],
        "notes": [
            "Train/val come from dataset train split.",
            "Heldout comes from dataset test split and should not be used for training.",
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

    dataset_cls = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    train_dataset = dataset_cls(root=str(download_dir), train=True, download=True)
    test_dataset = dataset_cls(root=str(download_dir), train=False, download=True)

    included_labels: Optional[List[str]] = None
    if args.include_labels.strip():
        included_labels = [x.strip() for x in args.include_labels.split(",") if x.strip()]
        unknown = [x for x in included_labels if x not in train_dataset.class_to_idx]
        if unknown:
            raise ValueError(
                f"Unknown labels for {args.dataset}: {unknown}. "
                f"Available labels include: {train_dataset.classes[:20]}"
            )
        allowed_ids = {train_dataset.class_to_idx[x] for x in included_labels}
        train_candidates = [i for i, y in enumerate(train_dataset.targets) if y in allowed_ids]
        test_candidates = [i for i, y in enumerate(test_dataset.targets) if y in allowed_ids]
    else:
        train_candidates = list(range(len(train_dataset)))
        test_candidates = list(range(len(test_dataset)))

    train_total_needed = args.train_size + args.val_size
    train_pool = _sample_indices(train_candidates, train_total_needed, rng)
    train_indices = train_pool[: args.train_size]
    val_indices = train_pool[args.train_size :]
    heldout_indices = _sample_indices(test_candidates, args.heldout_size, rng)

    print(f"Preparing split: train ({len(train_indices)} samples)")
    train_manifest = _write_split("train", train_dataset, train_indices, output_dir)
    print(f"Preparing split: val ({len(val_indices)} samples)")
    val_manifest = _write_split("val", train_dataset, val_indices, output_dir)
    print(f"Preparing split: heldout ({len(heldout_indices)} samples)")
    heldout_manifest = _write_split("heldout", test_dataset, heldout_indices, output_dir)

    _summarize(
        output_dir=output_dir,
        dataset_name=args.dataset,
        class_names=included_labels if included_labels else train_dataset.classes,
        train_size=len(train_indices),
        val_size=len(val_indices),
        heldout_size=len(heldout_indices),
        seed=args.seed,
        included_labels=included_labels,
    )

    print("Dataset setup complete.")
    print(f"train manifest:   {train_manifest}")
    print(f"val manifest:     {val_manifest}")
    print(f"heldout manifest: {heldout_manifest}")


if __name__ == "__main__":
    main()
