from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    GPT2Config,
    GPT2LMHeadModel,
)


@dataclass
class NativeVisionComponents:
    tokenizer: AutoTokenizer
    image_processor: CLIPImageProcessor


class NativeVisionGPT2(nn.Module):
    """
    GPT-2 small with native visual cross-attention.

    Image tokens come from a vision encoder and are projected to GPT-2 hidden size.
    GPT-2 can consume these visual tokens either through:
    - decoder cross-attention (fusion_mode="cross_attn"), or
    - a shared token stream (fusion_mode="unified"), where visual embeddings are
      prepended to text embeddings inside one decoder input sequence.
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        lm_model_name: str = "gpt2",
        fusion_mode: str = "cross_attn",
        freeze_vision: bool = True,
        freeze_lm_backbone: bool = True,
        unfreeze_top_n_gpt2_blocks: int = 0,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"cross_attn", "unified"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.vision_model_name = vision_model_name
        self.lm_model_name = lm_model_name
        self.fusion_mode = fusion_mode

        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)

        lm_config = GPT2Config.from_pretrained(lm_model_name)
        lm_config.add_cross_attention = fusion_mode == "cross_attn"
        self.lm = GPT2LMHeadModel.from_pretrained(lm_model_name, config=lm_config)

        self.visual_projector = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, self.lm.config.n_embd),
            nn.GELU(),
            nn.LayerNorm(self.lm.config.n_embd),
        )

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_lm_backbone:
            self._freeze_lm_backbone()
            if unfreeze_top_n_gpt2_blocks > 0:
                self._unfreeze_top_blocks(unfreeze_top_n_gpt2_blocks)

    def _freeze_lm_backbone(self) -> None:
        for param in self.lm.parameters():
            param.requires_grad = False

        if self.fusion_mode == "cross_attn":
            # Keep cross-attention trainable for native multimodal learning.
            for name, param in self.lm.named_parameters():
                if "crossattention" in name or "ln_cross_attn" in name:
                    param.requires_grad = True

    def _unfreeze_top_blocks(self, n: int) -> None:
        blocks = self.lm.transformer.h
        if n <= 0:
            return
        n = min(n, len(blocks))
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_tokens = vision_outputs.last_hidden_state
        return self.visual_projector(vision_tokens)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        visual_tokens = self.encode_images(pixel_values)
        visual_mask = torch.ones(
            visual_tokens.shape[:2],
            dtype=torch.long,
            device=visual_tokens.device,
        )

        if self.fusion_mode == "cross_attn":
            return self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_hidden_states=visual_tokens,
                encoder_attention_mask=visual_mask,
            )

        text_embeds = self.lm.transformer.wte(input_ids)
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            visual_ignore = torch.full(
                visual_tokens.shape[:2],
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([visual_ignore, labels], dim=1)

        return self.lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        visual_tokens = self.encode_images(pixel_values)
        visual_mask = torch.ones(
            visual_tokens.shape[:2],
            dtype=torch.long,
            device=visual_tokens.device,
        )

        if self.fusion_mode == "cross_attn":
            return self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=visual_tokens,
                encoder_attention_mask=visual_mask,
                **generate_kwargs,
            )

        # Unified mode generation: greedy decode over a single shared visual+text stream.
        # Returns only text token ids (prompt + generated), matching GPT-2 generate interface used by callers.
        max_new_tokens = int(generate_kwargs.get("max_new_tokens", 16))
        eos_token_id = generate_kwargs.get("eos_token_id", self.lm.config.eos_token_id)
        do_sample = bool(generate_kwargs.get("do_sample", False))
        temperature = float(generate_kwargs.get("temperature", 1.0))

        generated = input_ids
        generated_mask = attention_mask

        for _ in range(max_new_tokens):
            text_embeds = self.lm.transformer.wte(generated)
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            combined_mask = torch.cat([visual_mask, generated_mask], dim=1)

            outputs = self.lm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                use_cache=False,
            )
            next_logits = outputs.logits[:, -1, :]

            if do_sample:
                if temperature > 0:
                    next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            generated_mask = torch.cat(
                [
                    generated_mask,
                    torch.ones(
                        (generated_mask.shape[0], 1),
                        dtype=generated_mask.dtype,
                        device=generated_mask.device,
                    ),
                ],
                dim=1,
            )

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated


def load_processors(
    lm_model_name: str = "gpt2",
    vision_model_name: str = "openai/clip-vit-base-patch32",
) -> NativeVisionComponents:
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    return NativeVisionComponents(tokenizer=tokenizer, image_processor=image_processor)
