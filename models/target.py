"""
Target model wrapper: ViT + Projector + LLM (~3B, Qwen2-VL based).

During distillation the target is frozen; it acts as a teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


class TargetModel(nn.Module):
    """Thin wrapper around Qwen2-VL that exposes sub-components for drafter reuse."""

    def __init__(self, model_name_or_path: str, torch_dtype=torch.bfloat16):
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )
        self._freeze()

    def _freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Sub-component accessors used by drafter architectures               #
    # ------------------------------------------------------------------ #

    @property
    def visual(self):
        """Full Qwen2-VL visual encoder (ViT + merger/projector)."""
        return self.model.visual

    @property
    def embed_tokens(self):
        return self.model.model.embed_tokens

    @property
    def lm_hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def image_token_id(self) -> int:
        return self.model.config.image_token_id

    # ------------------------------------------------------------------ #
    # Feature extraction helpers                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_visual_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns merged (post-projector) visual features.

        Shape: (total_image_tokens, lm_hidden_size)
        These are already in the LLM's embedding space.
        """
        return self.model.visual(pixel_values, grid_thw=image_grid_thw)

    @torch.no_grad()
    def get_raw_vit_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw ViT features *before* the merger/projector.

        Used by Arch2 which shares the target ViT but uses its own projector.
        Shape: (total_patches, vit_hidden_size)
        """
        vis = self.model.visual
        hidden = vis.patch_embed(pixel_values)

        rotary_pos_emb = vis.rot_pos_emb(image_grid_thw)
        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        for blk in vis.blocks:
            hidden = blk(hidden, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return hidden  # (total_patches, vit_hidden_size)

    @property
    def vit_hidden_size(self) -> int:
        """Output dimension of the raw ViT (before merger)."""
        return self.model.visual.blocks[-1].norm2.normalized_shape[0]

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
