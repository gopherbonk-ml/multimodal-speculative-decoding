"""
Architecture 2: ViT(Target) + Projector(Drafter) + LLM(Drafter)

The target's raw ViT features (before the target's merger/projector) are reused.
The drafter trains its own MLP projector and its own small LLM.

This tests whether learning a task-specific projection from frozen vision features
is better than reusing the target's projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_drafter import BaseDrafter
from ..components.projector import MLPProjector
from ..target import TargetModel
from .small_llm_config import get_small_qwen2_config


class Arch2Drafter(BaseDrafter):
    """
    Frozen target ViT (raw, pre-merger features) + trainable projector + trainable small LLM.
    """

    def __init__(
        self,
        target: TargetModel,
        drafter_config: Qwen2Config | None = None,
        pool_factor: int = 1,
        projector_dropout: float = 0.0,
    ):
        super().__init__()

        # Store reference to target for raw feature extraction (no params here)
        self._target = target
        self._image_token_id = target.image_token_id

        vit_dim = target.vit_hidden_size

        if drafter_config is None:
            drafter_config = get_small_qwen2_config(target.vocab_size)

        drafter_dim = drafter_config.hidden_size

        # Trainable
        self.projector = MLPProjector(
            in_dim=vit_dim,
            out_dim=drafter_dim,
            pool_factor=pool_factor,
            dropout=projector_dropout,
        )
        self.llm = Qwen2ForCausalLM(drafter_config)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.llm.model.embed_tokens

    @property
    def image_token_id(self) -> int:
        return self._image_token_id

    def prepare_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract raw ViT features from target (no grad), then project with drafter projector."""
        raw = self._target.get_raw_vit_features(pixel_values, image_grid_thw)  # (N, vit_dim), no grad
        return self.projector(raw)  # (N', drafter_dim), grad flows through projector

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None and pixel_values is not None:
            visual_embeds = self.prepare_vision(pixel_values, image_grid_thw)
            inputs_embeds = self.build_inputs_embeds(
                input_ids, visual_embeds, self._image_token_id
            )
            input_ids = None

        return self.llm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
