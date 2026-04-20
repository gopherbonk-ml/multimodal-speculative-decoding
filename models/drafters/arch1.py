"""
Architecture 1: ViT(Target) + Projector(Target) + Adapter(Drafter) + LLM(Drafter)

The target's visual encoder (ViT + merger) is shared and frozen.
A lightweight adapter maps the target LLM dim → drafter LLM dim.
Only the adapter and the small LLM are trained.

This is the most "parasitic" architecture — maximally reuses the target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_drafter import BaseDrafter
from ..components.adapter import VisualAdapter
from ..target import TargetModel
from .small_llm_config import get_small_qwen2_config


class Arch1Drafter(BaseDrafter):
    """
    Uses frozen target visual encoder (ViT + projector) and trains:
      - VisualAdapter   : target_dim → drafter_dim
      - Small Qwen2 LLM : ~88M params
    """

    def __init__(
        self,
        target: TargetModel,
        drafter_config: Qwen2Config | None = None,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()

        # Frozen target visual encoder (ViT + built-in projector)
        self.target_visual = target.visual
        for p in self.target_visual.parameters():
            p.requires_grad = False

        self._image_token_id = target.image_token_id

        target_dim = target.lm_hidden_size

        if drafter_config is None:
            drafter_config = get_small_qwen2_config(target.vocab_size)

        drafter_dim = drafter_config.hidden_size

        # Trainable components
        self.adapter = VisualAdapter(target_dim, drafter_dim, dropout=adapter_dropout)
        self.llm = Qwen2ForCausalLM(drafter_config)

    # ------------------------------------------------------------------ #
    # BaseDrafter interface                                               #
    # ------------------------------------------------------------------ #

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.llm.model.embed_tokens

    @property
    def image_token_id(self) -> int:
        return self._image_token_id

    @torch.no_grad()
    def prepare_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Target ViT + projector → adapter → drafter space."""
        visual = self.target_visual(pixel_values, grid_thw=image_grid_thw)  # (N, target_dim)
        return self.adapter(visual)  # (N, drafter_dim)

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
        if inputs_embeds is None:
            if pixel_values is not None:
                # prepare_vision is not wrapped in no_grad here so adapter gets gradients
                with torch.no_grad():
                    visual_raw = self.target_visual(pixel_values, grid_thw=image_grid_thw)
                visual_embeds = self.adapter(visual_raw)  # gradients flow through adapter
                inputs_embeds = self.build_inputs_embeds(
                    input_ids, visual_embeds, self._image_token_id
                )
                input_ids = None  # model uses inputs_embeds
            # else: text-only batch, let the LLM handle input_ids normally

        return self.llm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
