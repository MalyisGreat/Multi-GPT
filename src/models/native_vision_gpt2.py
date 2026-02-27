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
    GPT-2 attends to these visual tokens through decoder cross-attention.
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        lm_model_name: str = "gpt2",
        freeze_vision: bool = True,
        freeze_lm_backbone: bool = True,
        unfreeze_top_n_gpt2_blocks: int = 0,
    ) -> None:
        super().__init__()

        self.vision_model_name = vision_model_name
        self.lm_model_name = lm_model_name

        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)

        lm_config = GPT2Config.from_pretrained(lm_model_name)
        lm_config.add_cross_attention = True
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
        encoder_hidden_states = self.encode_images(pixel_values)
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        encoder_hidden_states = self.encode_images(pixel_values)
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        return self.lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **generate_kwargs,
        )


def load_processors(
    lm_model_name: str = "gpt2",
    vision_model_name: str = "openai/clip-vit-base-patch32",
) -> NativeVisionComponents:
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    return NativeVisionComponents(tokenizer=tokenizer, image_processor=image_processor)
