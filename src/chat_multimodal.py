from __future__ import annotations

import argparse
import collections
import inspect
import json
import re
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image


CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR100_LABELS = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer_keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

# Kept intentionally small and conservative for this checkpoint.
EDIBLE_LABELS = {
    "apple",
    "orange",
    "pear",
    "sweet_pepper",
    "mushroom",
}

NON_FOOD_HINT_LABELS = {
    "airplane",
    "rocket",
    "truck",
    "automobile",
    "bicycle",
    "ship",
    "train",
    "bus",
    "tank",
}


def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label_key(label: str) -> str:
    return _normalize_text(label).replace(" ", "_")


def _humanize_label(label: str) -> str:
    return label.replace("_", " ")


def _indefinite_article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _dedupe_sentence_chunks(text: str) -> str:
    chunks = [c.strip() for c in re.split(r"[.!?]+", text) if c.strip()]
    deduped: List[str] = []
    seen: Set[str] = set()
    for chunk in chunks:
        key = _normalize_text(chunk)
        if key and key not in seen:
            deduped.append(chunk)
            seen.add(key)
    return ". ".join(deduped).strip()


def _detect_intent(question: str) -> str:
    q = question.lower().strip()
    if re.search(r"\b(eat|edible|food|safe to eat|can i eat|could i eat)\b", q):
        return "edibility"
    if re.search(r"\b(what is this|what's this|what is in this|identify|name this)\b", q):
        return "identify"
    if re.search(r"\b(describe|caption)\b", q):
        return "describe"
    return "general"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a multimodal GPT-2 checkpoint (image + question)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Checkpoint directory with run_config.json and epoch-*.pt files.",
    )
    parser.add_argument(
        "--checkpoint-archive",
        type=str,
        default="",
        help="Optional .tar.gz archive to auto-extract (for example unified-cifar100-long-best.tar.gz).",
    )
    parser.add_argument(
        "--extract-root",
        type=str,
        default=".",
        help="Where to extract the checkpoint archive.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Explicit checkpoint file path. If empty, latest epoch-*.pt from checkpoint-dir is used.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--port-tries",
        type=int,
        default=20,
        help="If --port is busy, try this many sequential ports before failing.",
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--smoke-test-image",
        type=str,
        default="",
        help="Run one inference on this image path and exit (no web UI).",
    )
    parser.add_argument(
        "--smoke-test-question",
        type=str,
        default="what is this?",
        help="Question used for smoke test mode.",
    )
    return parser.parse_args()


def _safe_extract_tar(archive_path: Path, extract_root: Path) -> List[Path]:
    extract_root = extract_root.resolve()
    checkpoint_dirs: List[Path] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target_path = (extract_root / member.name).resolve()
            if not str(target_path).startswith(str(extract_root)):
                raise RuntimeError(f"Blocked unsafe archive member path: {member.name}")

        if "filter" in inspect.signature(tar.extractall).parameters:
            tar.extractall(path=extract_root, filter="data")
        else:
            tar.extractall(path=extract_root)

        for member in members:
            if member.name.endswith("run_config.json"):
                checkpoint_dirs.append((extract_root / Path(member.name).parent).resolve())
            if re.search(r"epoch-\d+\.pt$", member.name):
                checkpoint_dirs.append((extract_root / Path(member.name).parent).resolve())

    unique_dirs = sorted(set(checkpoint_dirs), key=lambda p: len(str(p)), reverse=True)
    return unique_dirs


def _list_checkpoint_dirs_in_archive(archive_path: Path) -> List[Path]:
    checkpoint_dirs: List[Path] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("run_config.json"):
                checkpoint_dirs.append(Path(member.name).parent)
            if re.search(r"epoch-\d+\.pt$", member.name):
                checkpoint_dirs.append(Path(member.name).parent)
    return sorted(set(checkpoint_dirs), key=lambda p: len(str(p)), reverse=True)


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("epoch-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    def epoch_num(path: Path) -> int:
        m = re.search(r"epoch-(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=epoch_num)[-1]


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"--checkpoint-dir does not exist: {checkpoint_dir}")
        return checkpoint_dir

    if args.checkpoint_archive:
        archive_path = Path(args.checkpoint_archive).resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"--checkpoint-archive does not exist: {archive_path}")
        extract_root = Path(args.extract_root).resolve()
        extract_root.mkdir(parents=True, exist_ok=True)

        archive_checkpoint_dirs = _list_checkpoint_dirs_in_archive(archive_path)
        for rel_dir in archive_checkpoint_dirs:
            abs_dir = (extract_root / rel_dir).resolve()
            if (abs_dir / "run_config.json").exists():
                return abs_dir

        extracted_dirs = _safe_extract_tar(archive_path, extract_root)
        if not extracted_dirs:
            raise RuntimeError(
                f"Could not infer checkpoint directory from archive members: {archive_path}"
            )
        return extracted_dirs[0]

    raise ValueError("Provide either --checkpoint-dir or --checkpoint-archive.")


class MultimodalChatModel:
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_path: Optional[Path],
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> None:
        # Lazy import so users see startup logs before heavy HF imports.
        from models.native_vision_gpt2 import NativeVisionGPT2, load_processors

        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        run_config_path = checkpoint_dir / "run_config.json"
        run_config = {}
        if run_config_path.exists():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        self.run_config = run_config

        vision_model_name = run_config.get("vision_model_name", "openai/clip-vit-base-patch32")
        lm_model_name = run_config.get("lm_model_name", "gpt2")
        fusion_mode = run_config.get("fusion_mode", "cross_attn")
        unfreeze_top_n = int(run_config.get("unfreeze_top_n_blocks", 0))

        self.model = NativeVisionGPT2(
            vision_model_name=vision_model_name,
            lm_model_name=lm_model_name,
            fusion_mode=fusion_mode,
            freeze_vision=not bool(run_config.get("no_freeze_vision", False)),
            freeze_lm_backbone=not bool(run_config.get("no_freeze_lm_backbone", False)),
            unfreeze_top_n_gpt2_blocks=unfreeze_top_n,
        ).to(device)

        if checkpoint_path is None:
            checkpoint_path = _latest_epoch_checkpoint(checkpoint_dir)
        self.checkpoint_path = checkpoint_path.resolve()

        ckpt = torch.load(self.checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys while loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys while loading checkpoint: {len(unexpected)}")

        components = load_processors(
            lm_model_name=lm_model_name,
            vision_model_name=vision_model_name,
        )
        self.tokenizer = components.tokenizer
        self.image_processor = components.image_processor

        self.tokenizer.padding_side = "left"
        self.model.lm.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.lm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.label_vocab = self._load_label_vocab()
        self.label_aliases = self._build_label_aliases(self.label_vocab)

    def _build_label_aliases(self, labels: Sequence[str]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for label in labels:
            canonical = _normalize_label_key(label)
            if canonical:
                aliases[_normalize_text(label)] = canonical
                aliases[_normalize_text(_humanize_label(label))] = canonical
        return aliases

    def _load_label_vocab(self) -> List[str]:
        labels: Set[str] = set()
        manifest_candidates = []
        for key in ("train_jsonl", "val_jsonl", "heldout_jsonl"):
            value = self.run_config.get(key)
            if isinstance(value, str) and value.strip():
                manifest_candidates.append(value.strip())

        for manifest_str in manifest_candidates:
            manifest_path = Path(manifest_str)
            if not manifest_path.is_absolute():
                manifest_path = (Path.cwd() / manifest_path).resolve()
            if not manifest_path.exists():
                continue

            summary_path = manifest_path.parent / "dataset_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    class_names = summary.get("class_names")
                    if isinstance(class_names, list):
                        for name in class_names:
                            if isinstance(name, str) and name.strip():
                                labels.add(_normalize_label_key(name))
                except Exception:
                    pass

            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        if line_idx >= 50000:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue
                        label_value = record.get("label")
                        if isinstance(label_value, str) and label_value.strip():
                            labels.add(_normalize_label_key(label_value))
            except Exception:
                continue

        if labels:
            return sorted(labels)

        # Fallback when manifests are not present locally.
        fallback_key = " ".join(
            [
                str(self.run_config.get("train_jsonl", "")),
                str(self.run_config.get("val_jsonl", "")),
                str(self.run_config.get("heldout_jsonl", "")),
                str(self.checkpoint_dir),
            ]
        ).lower()
        if "cifar100" in fallback_key:
            return sorted({_normalize_label_key(x) for x in CIFAR100_LABELS})
        return sorted({_normalize_label_key(x) for x in CIFAR10_LABELS})

    def _generate_raw(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        text_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = text_inputs["input_ids"].to(self.device)
        if input_ids.shape[1] == 0:
            input_ids = torch.tensor(
                [[self.tokenizer.eos_token_id]],
                dtype=torch.long,
                device=self.device,
            )
        attention_mask = torch.ones_like(input_ids)

        image_inputs = self.image_processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_new = generated[:, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_new[0], skip_special_tokens=True).strip()
        if not decoded:
            decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return _dedupe_sentence_chunks(decoded) or "(no output)"

    def _extract_label_from_text(self, text: str) -> Optional[str]:
        normalized = _normalize_text(text)
        if not normalized:
            return None

        hits: List[Tuple[int, int, str]] = []
        for alias, canonical in self.label_aliases.items():
            start = normalized.find(alias)
            if start >= 0:
                hits.append((start, len(alias), canonical))
        if hits:
            hits.sort(key=lambda x: (x[0], -x[1]))
            return hits[0][2]

        # Fallback heuristics when text is noisy/repetitive.
        fallback_patterns = [
            r"(?:photo|image|picture)\s+of\s+(?:a|an|the)?\s*([a-z][a-z\s_]{1,40})",
            r"^(?:a|an|the)\s+([a-z][a-z\s_]{1,40})",
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, normalized)
            if match:
                chunk = match.group(1).strip()
                chunk = re.sub(r"\b(in|on|at)\b.*$", "", chunk).strip()
                chunk = chunk.replace(" ", "_")
                if chunk:
                    return chunk

        return None

    def _predict_label(self, image: Image.Image) -> Tuple[Optional[str], str]:
        prompts = [
            "Question: what is in this image?\nAnswer:",
            "Question: identify the main object with a short label.\nAnswer:",
        ]
        labels: List[str] = []
        raw_outputs: List[str] = []

        for prompt in prompts:
            raw = self._generate_raw(image=image, prompt=prompt, max_new_tokens=16)
            raw_outputs.append(raw)
            label = self._extract_label_from_text(raw)
            if label:
                labels.append(label)

        if labels:
            best_label = collections.Counter(labels).most_common(1)[0][0]
            return best_label, raw_outputs[0]
        return None, raw_outputs[0] if raw_outputs else ""

    def answer(self, image: Image.Image, question: str) -> str:
        if image is None:
            return "Please upload an image."

        question_clean = question.strip() or "What is this image?"
        intent = _detect_intent(question_clean)

        label, raw_caption = self._predict_label(image=image)
        if label:
            human_label = _humanize_label(label)
            article = _indefinite_article(human_label)
        else:
            human_label = ""
            article = "a"

        if intent == "identify":
            if label:
                return f"It looks like {article} {human_label}."
            return f"I am not confident on the object. Model output: {raw_caption}"

        if intent == "describe":
            if label:
                return f"This appears to be {article} {human_label}."
            return raw_caption

        if intent == "edibility":
            if not label:
                return "I cannot determine the object confidently, so I cannot answer edibility."
            if label in EDIBLE_LABELS:
                return (
                    f"It looks like {article} {human_label}. "
                    f"That is generally edible, but this model cannot verify preparation or safety."
                )
            if label in NON_FOOD_HINT_LABELS:
                return f"It looks like {article} {human_label}. That is not food."
            return f"It looks like {article} {human_label}. This model is not reliable for safety/edibility decisions."

        if label:
            return (
                f"It looks like {article} {human_label}. "
                "This checkpoint is tuned for object identification and short captions."
            )
        return raw_caption


def run_ui(
    chat_model: MultimodalChatModel,
    host: str,
    port: int,
    share: bool,
    port_tries: int,
) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Gradio is required for UI mode. Install with: pip install gradio"
        ) from exc

    def ask(image_file: Any, question: str, history: Optional[List[Dict[str, Any]]]):
        history = list(history or [])
        image: Optional[Image.Image] = None
        load_error: Optional[str] = None

        if isinstance(image_file, (bytes, bytearray)):
            if len(image_file) == 0:
                load_error = "Uploaded file is empty (0 bytes). Re-save the image as PNG/JPG and upload again."
            else:
                try:
                    image = Image.open(BytesIO(image_file)).convert("RGB")
                except Exception as exc:
                    load_error = f"Could not decode uploaded image bytes: {exc}"
        else:
            image_path: Optional[str] = None
            if isinstance(image_file, str):
                image_path = image_file
            elif hasattr(image_file, "name"):
                image_path = str(image_file.name)

            if image_path:
                try:
                    file_size = Path(image_path).stat().st_size
                    if file_size == 0:
                        load_error = (
                            "Uploaded file is empty (0 bytes). Re-save the image as PNG/JPG and upload again."
                        )
                    else:
                        with Image.open(image_path) as opened:
                            image = opened.convert("RGB")
                except Exception as exc:
                    load_error = f"Could not read image file: {exc}"

        if image is None:
            answer = load_error or "Please upload an image file before asking."
        else:
            try:
                answer = chat_model.answer(image=image, question=question)
            except Exception as exc:
                answer = f"Inference error: {exc}"

        if image_file is None:
            answer = "Please upload an image file before asking."

        user_text = question.strip() if question and question.strip() else "[Describe image]"
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    with gr.Blocks(title="Multimodal GPT-2 Chat") as demo:
        gr.Markdown("## Multimodal GPT-2 Chat\nUpload an image, ask a question, and get a model answer.")
        with gr.Row():
            image_input = gr.File(label="Image File", file_types=["image"], type="binary")
            with gr.Column():
                question_input = gr.Textbox(label="Question", placeholder="What is in this image?")
                ask_btn = gr.Button("Ask")
                clear_btn = gr.Button("Clear Chat")
        chatbox = gr.Chatbot(label="Chat")

        ask_btn.click(
            ask,
            inputs=[image_input, question_input, chatbox],
            outputs=[chatbox, question_input],
        )
        question_input.submit(
            ask,
            inputs=[image_input, question_input, chatbox],
            outputs=[chatbox, question_input],
        )
        clear_btn.click(lambda: [], outputs=[chatbox], inputs=[])

    attempts = max(1, int(port_tries))
    launch_error: Optional[Exception] = None
    for attempt in range(attempts):
        candidate_port = port + attempt
        try:
            if attempt > 0:
                print(
                    f"Requested port {port} busy. Retrying on port {candidate_port}..."
                )
            demo.launch(server_name=host, server_port=candidate_port, share=share)
            return
        except OSError as exc:
            if "Cannot find empty port in range" not in str(exc):
                raise
            launch_error = exc

    if launch_error is not None:
        raise launch_error


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args)
    checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else None

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using checkpoint dir: {checkpoint_dir}")
    print(f"Using device: {device}")
    chat_model = MultimodalChatModel(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(f"Loaded checkpoint: {chat_model.checkpoint_path}")

    if args.smoke_test_image:
        image_path = Path(args.smoke_test_image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Smoke test image not found: {image_path}")
        with Image.open(image_path) as img:
            answer = chat_model.answer(img.convert("RGB"), args.smoke_test_question)
        print(json.dumps({"question": args.smoke_test_question, "answer": answer}, indent=2))
        return

    run_ui(
        chat_model,
        host=args.host,
        port=args.port,
        share=args.share,
        port_tries=args.port_tries,
    )


if __name__ == "__main__":
    main()
