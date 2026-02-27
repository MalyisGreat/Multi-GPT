from __future__ import annotations

import argparse
import inspect
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

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

    def answer(self, image: Image.Image, question: str) -> str:
        if image is None:
            return "Please upload an image."

        prompt = question.strip()
        if not prompt:
            prompt = "Describe this image."
        prompt = f"Question: {prompt}\nAnswer:"

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
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_new = generated[:, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_new[0], skip_special_tokens=True).strip()
        if not decoded:
            decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return decoded or "(no output)"


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
        image_path: Optional[str] = None
        if isinstance(image_file, str):
            image_path = image_file
        elif hasattr(image_file, "name"):
            image_path = str(image_file.name)

        if not image_path:
            answer = "Please upload an image file before asking."
        else:
            try:
                with Image.open(image_path) as image:
                    answer = chat_model.answer(image=image.convert("RGB"), question=question)
            except Exception as exc:
                answer = f"Could not read image file: {exc}"

        user_text = question.strip() if question and question.strip() else "[Describe image]"
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    with gr.Blocks(title="Multimodal GPT-2 Chat") as demo:
        gr.Markdown("## Multimodal GPT-2 Chat\nUpload an image, ask a question, and get a model answer.")
        with gr.Row():
            image_input = gr.File(label="Image File", file_types=["image"], type="filepath")
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
