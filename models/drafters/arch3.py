"""
Architecture 3: ViT(Drafter) + Projector(Drafter) + LLM(Drafter)

Fully independent drafter. All components are trained from scratch (or distilled).
No parameters are shared with the target model.

This is the most self-contained drafter — the comparison baseline for
"how much does sharing vision components help?"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.base_drafter import BaseDrafter
from models.components.small_vit import SmallViT
from models.components.projector import MLPProjector
from models.target import TargetModel
from .small_llm_config import get_small_qwen2_config


class Arch3Drafter(BaseDrafter):
    """
    Fully independent drafter: small ViT + projector + small LLM.
    All weights are trainable.
    """

    def __init__(
        self,
        target: TargetModel,
        drafter_config: Qwen2Config | None = None,
        # SmallViT hyper-parameters
        img_size: int = 224,
        patch_size: int = 14,
        vit_embed_dim: int = 384,
        vit_depth: int = 8,
        vit_num_heads: int = 6,
        pool_factor: int = 1,
        projector_dropout: float = 0.0,
        vit_dropout: float = 0.0,
    ):
        super().__init__()

        self._image_token_id = target.image_token_id

        if drafter_config is None:
            drafter_config = get_small_qwen2_config(target.vocab_size)

        drafter_dim = drafter_config.hidden_size

        # All trainable
        self.vit = SmallViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            dropout=vit_dropout,
        )
        self.projector = MLPProjector(
            in_dim=vit_embed_dim,
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
        """Small ViT → projector → drafter LLM dim."""
        # pixel_values expected shape: (B, C, H, W)
        # SmallViT returns (B*N, vit_embed_dim)
        raw = self.vit(pixel_values)
        return self.projector(raw)  # (B*N', drafter_dim)

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
