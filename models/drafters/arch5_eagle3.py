"""
Architecture 5 (EAGLE-3): Feature-conditioned multimodal drafter.

Core idea (Chen et al., 2024 — EAGLE / EAGLE-3):
    At each position t, the drafter input is NOT the raw token embedding,
    but a fusion of (projected_h_{t-1}, embed(x_t)), where h_{t-1} is:
        - Training / target-prefill : target's last-layer hidden state at t-1
        - Draft generation          : drafter's own last-layer hidden state at t-1

    This makes the drafter "representation-calibrated" to the target and
    significantly boosts acceptance rate without extra target forward calls.

Visual tokens:
    Same strategy as Arch1: frozen target ViT+merger, lightweight VisualAdapter
    maps the result to the drafter's hidden dimension before fusion.

Training loss (Eagle3Loss):
    alpha * KL(target || drafter)  +  beta * MSE_feat  +  (1-alpha-beta) * CE

References:
    EAGLE  : https://arxiv.org/abs/2401.15077
    EAGLE-3: https://arxiv.org/abs/2503.01840
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.base_drafter import BaseDrafter
from models.components.adapter import VisualAdapter
from models.target import TargetModel
from .small_llm_config import get_small_qwen2_config


class FeatureFusion(nn.Module):
    """
    Fuses projected feature h_{t-1} with token embedding e_t.

    Input : cat([projected_h, token_embed])  — (..., feature_dim + embed_dim)
    Output: drafter LLM input embedding      — (..., drafter_dim)

    Two-layer SiLU MLP followed by LayerNorm.
    """

    def __init__(self, feature_dim: int, embed_dim: int, drafter_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, drafter_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(drafter_dim * 2, drafter_dim, bias=False),
        )
        self.norm = nn.LayerNorm(drafter_dim)

    def forward(self, projected_h: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projected_h:  (B, L, feature_dim)  — already-projected hidden state
            token_embeds: (B, L, embed_dim)     — raw token / visual embeddings
        Returns:
            (B, L, drafter_dim)
        """
        return self.norm(self.net(torch.cat([projected_h, token_embeds], dim=-1)))


class Arch5Eagle3Drafter(BaseDrafter):
    """
    EAGLE-3 multimodal drafter (Architecture 5).

    Trainable components:
        target_feat_proj  — Linear(target_dim  → feature_dim) + LN
        drafter_feat_proj — Linear(drafter_dim → feature_dim) + LN
        feature_fusion    — FeatureFusion(feature_dim, embed_dim → drafter_dim)
        visual_adapter    — VisualAdapter(target_dim → drafter_dim)
        llm               — small Qwen2ForCausalLM (~88M)

    Frozen:
        target_visual     — target ViT + merger (Qwen2-VL)
    """

    def __init__(
        self,
        target: TargetModel,
        drafter_config: Qwen2Config | None = None,
        # feature_dim defaults to drafter hidden_size so feat-align is in the same space
        feature_dim: int | None = None,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()

        # --- frozen target visual encoder ---
        self.target_visual = target.visual
        for p in self.target_visual.parameters():
            p.requires_grad = False

        self._image_token_id = target.image_token_id
        target_hidden_dim = target.lm_hidden_size

        if drafter_config is None:
            drafter_config = get_small_qwen2_config(target.vocab_size)

        drafter_dim = drafter_config.hidden_size
        embed_dim = drafter_dim  # Qwen2: embed_tokens output dim == hidden_size

        if feature_dim is None:
            feature_dim = drafter_dim

        self._target_hidden_dim = target_hidden_dim
        self._feature_dim = feature_dim
        self._drafter_dim = drafter_dim

        # --- trainable ---
        self.target_feat_proj = nn.Sequential(
            nn.Linear(target_hidden_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
        )
        # Maps drafter hidden → same feature space so we can re-use FeatureFusion at inference
        self.drafter_feat_proj = nn.Sequential(
            nn.Linear(drafter_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
        )
        self.feature_fusion = FeatureFusion(feature_dim, embed_dim, drafter_dim)
        self.visual_adapter = VisualAdapter(target_hidden_dim, drafter_dim, dropout=adapter_dropout)
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
        """Frozen target ViT → VisualAdapter → drafter dim."""
        visual = self.target_visual(pixel_values, grid_thw=image_grid_thw)
        return self.visual_adapter(visual)  # (N_patches, drafter_dim)

    # ------------------------------------------------------------------ #
    # Feature projection helpers                                          #
    #   Called externally by Eagle3Trainer / Eagle3SpeculativeDecoder     #
    # ------------------------------------------------------------------ #

    def project_target_features(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """(B, L, target_dim) → (B, L, feature_dim)  [no grad flows into target_hidden]."""
        return self.target_feat_proj(target_hidden)

    def project_drafter_features(self, drafter_hidden: torch.Tensor) -> torch.Tensor:
        """(B, *, drafter_dim) → (B, *, feature_dim)."""
        return self.drafter_feat_proj(drafter_hidden)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
        inputs_embeds: torch.Tensor | None = None,
        projected_features: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            projected_features: (B, L, feature_dim) — already-projected h_{t-1}.
                - Training    : project_target_features(target_hidden_shifted)
                - Inference   : project_drafter_features(drafter_hidden_prev)
                - None        : treated as all-zeros (no feature conditioning)
            output_hidden_states: always True for Eagle3; needed by Eagle3Loss and
                                  Eagle3SpeculativeDecoder to extract h_{t} for next step.
        """
        if inputs_embeds is None:
            # 1. Build token embeddings (with visual injection for image tokens)
            if pixel_values is not None:
                with torch.no_grad():
                    visual_raw = self.target_visual(pixel_values, grid_thw=image_grid_thw)
                visual_embeds = self.visual_adapter(visual_raw)
                token_embeds = self.build_inputs_embeds(
                    input_ids, visual_embeds, self._image_token_id
                )
            else:
                token_embeds = self.embed_tokens(input_ids)  # (B, L, drafter_dim)

            # 2. Prepare feature tensor (zeros if not provided)
            B, L = token_embeds.shape[:2]
            if projected_features is None:
                projected_features = torch.zeros(
                    B, L, self._feature_dim,
                    device=token_embeds.device, dtype=token_embeds.dtype,
                )

            # 3. Fuse: (B, L, feature_dim) + (B, L, drafter_dim) → (B, L, drafter_dim)
            inputs_embeds = self.feature_fusion(projected_features, token_embeds)

        return self.llm(
            input_ids=None,          # always bypassed; we pass inputs_embeds
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
