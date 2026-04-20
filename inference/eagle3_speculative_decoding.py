"""
EAGLE-3 speculative decoding.

Extends SpeculativeDecoder to manage the feature hidden states that
Arch5Eagle3Drafter requires at each draft step.

Prefill:
    Both target and drafter process the prompt.
    The drafter uses projected target hidden states as feature conditioning.
    The drafter's last hidden state is saved as the seed for draft generation.

Draft (γ steps):
    At each step i the drafter input is:
        fused = FeatureFusion(project_drafter_features(h'_{t-1}), embed(x_t))
    where h'_{t-1} is the drafter's own hidden state from the previous step.
    This avoids any additional target calls during drafting.

Verify:
    Identical to the base SpeculativeDecoder (single target forward pass).

Hidden state management after acceptance:
    After verifying and accepting n tokens, we roll the drafter KV cache back
    as usual, and set the feature hidden to all_hidden[n_accepted] — the
    drafter's hidden at the last accepted position.  This is the correct seed
    for the next draft round.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F

from .speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    DecodingStats,
)
from ..models.target import TargetModel
from ..models.drafters.arch5_eagle3 import Arch5Eagle3Drafter


class Eagle3SpeculativeDecoder(SpeculativeDecoder):
    """
    Speculative decoding for Arch5Eagle3Drafter.

    The public interface is identical to SpeculativeDecoder.generate().
    """

    def __init__(
        self,
        target: TargetModel,
        drafter: Arch5Eagle3Drafter,
        config: SpeculativeDecodingConfig | None = None,
    ):
        super().__init__(target, drafter, config)
        # Convenience accessor that bypasses DDP wrapper if present
        self._drafter_module: Arch5Eagle3Drafter = (
            drafter.module if hasattr(drafter, "module") else drafter
        )

    # ------------------------------------------------------------------ #
    # Main entry point (overrides parent)                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, DecodingStats]:
        assert input_ids.shape[0] == 1, "Batch size must be 1 for speculative decoding"

        stats = DecodingStats()
        t_start = time.perf_counter()

        # ---- Prefill ----
        target_past, drafter_past, prev_drafter_hidden = self._eagle3_prefill(
            input_ids, pixel_values, image_grid_thw, attention_mask
        )
        # prev_drafter_hidden: (1, 1, drafter_dim) — drafter hidden at prefix[-1]

        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < self.cfg.max_new_tokens:
            if self._is_done(generated):
                break

            # ---- Draft ----
            draft_tokens, draft_log_probs, drafter_past, all_hidden = self._eagle3_draft(
                generated, drafter_past, prev_drafter_hidden
            )
            # all_hidden[i] = drafter hidden AFTER processing draft token i-1
            # (all_hidden[0] = prev_drafter_hidden passed in)

            # ---- Verify ----
            accepted_tokens, bonus_token, target_past = self._verify(
                generated, draft_tokens, draft_log_probs, target_past
            )

            n_accepted = accepted_tokens.shape[1]
            stats.total_draft_tokens += len(draft_tokens)
            stats.total_accepted_tokens += n_accepted + 1
            stats.total_target_calls += 1

            new_tokens = torch.cat([accepted_tokens, bonus_token], dim=1)
            generated = torch.cat([generated, new_tokens], dim=1)
            stats.total_tokens_generated += new_tokens.shape[1]

            # ---- Rollback drafter KV cache ----
            drafter_past = self._rollback_drafter_cache(drafter_past, n_accepted + 1)

            # ---- Update feature hidden for next round ----
            # all_hidden[n_accepted] is the drafter's hidden at the last accepted position.
            # The next draft round feeds bonus_token as context and uses this as feature seed.
            prev_drafter_hidden = all_hidden[n_accepted]  # (1, 1, drafter_dim)

        stats.wall_time_seconds = time.perf_counter() - t_start
        return generated, stats

    # ------------------------------------------------------------------ #
    # EAGLE-3 prefill                                                     #
    # ------------------------------------------------------------------ #

    def _eagle3_prefill(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        """
        Run target and drafter on the prompt; return KV caches and the
        drafter's hidden state at the last prefix position (for draft seeding).

        Returns:
            target_past         — target KV cache
            drafter_past        — drafter KV cache
            drafter_last_hidden — (1, 1, drafter_dim)  feature seed for first draft step
        """
        prefix = input_ids[:, :-1]
        prefix_mask = attention_mask[:, :-1] if attention_mask is not None else None

        # ---- Target prefill with hidden states ----
        target_out = self.target(
            input_ids=prefix,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=prefix_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        target_past = target_out.past_key_values
        target_hidden = target_out.hidden_states[-1]  # (1, L-1, target_dim)

        # ---- Build shifted & projected features for drafter prefill ----
        # At position t, drafter receives project_target_features(h_{t-1}).
        # Position 0 gets a zero vector.
        B, L_prefix, target_dim = target_hidden.shape
        device, dtype = target_hidden.device, target_hidden.dtype

        h_shifted_raw = torch.cat([
            torch.zeros(B, 1, target_dim, device=device, dtype=dtype),
            target_hidden[:, :-1, :],
        ], dim=1)  # (1, L-1, target_dim) — h_{t-1} at each position

        projected_features = self._drafter_module.project_target_features(h_shifted_raw)
        # (1, L-1, feature_dim)

        # ---- Drafter prefill ----
        drafter_out = self.drafter(
            input_ids=prefix,
            attention_mask=prefix_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            projected_features=projected_features,
            use_cache=True,
            output_hidden_states=True,
        )
        drafter_past = drafter_out.past_key_values

        # Last hidden state of the drafter after the prefix:
        # This is h'_{L-2} (0-indexed), i.e. the hidden at the final prefix token.
        # It will be used to condition generation of draft token at position L-1.
        drafter_last_hidden = drafter_out.hidden_states[-1][:, -1:, :]  # (1, 1, drafter_dim)

        return target_past, drafter_past, drafter_last_hidden

    # ------------------------------------------------------------------ #
    # EAGLE-3 draft                                                       #
    # ------------------------------------------------------------------ #

    def _eagle3_draft(
        self,
        context: torch.LongTensor,
        drafter_past,
        prev_drafter_hidden: torch.Tensor,
    ) -> tuple[list, list, object, list]:
        """
        Auto-regressively generate γ tokens with feature conditioning.

        Args:
            context:            current generated sequence (1, L)
            drafter_past:       drafter KV cache
            prev_drafter_hidden: (1, 1, drafter_dim) — drafter hidden at context[-1]

        Returns:
            draft_tokens:    list[Tensor(1,1)] length γ
            draft_log_probs: list[Tensor(1,V)] length γ
            drafter_past:    updated KV cache
            all_hidden:      list[Tensor(1,1,drafter_dim)] length γ+1
                             all_hidden[0] = prev_drafter_hidden (input seed)
                             all_hidden[i] = drafter hidden after draft token i-1
        """
        draft_tokens: list = []
        draft_log_probs: list = []
        all_hidden: list = [prev_drafter_hidden]

        last_token = context[:, -1:]  # (1, 1)
        cur_hidden = prev_drafter_hidden  # (1, 1, drafter_dim)

        for _ in range(self.cfg.gamma):
            # Project drafter's previous hidden to feature space
            projected = self._drafter_module.project_drafter_features(cur_hidden)
            # projected: (1, 1, feature_dim)

            out = self.drafter(
                input_ids=last_token,
                pixel_values=None,       # image already in KV cache
                image_grid_thw=None,
                projected_features=projected,
                use_cache=True,
                past_key_values=drafter_past,
                output_hidden_states=True,
            )

            logits = out.logits[:, -1, :]   # (1, V)
            log_probs = F.log_softmax(logits / max(self.cfg.temperature, 1e-6), dim=-1)
            token = self._sample(log_probs)  # (1, 1)

            draft_tokens.append(token)
            draft_log_probs.append(log_probs)

            drafter_past = out.past_key_values
            cur_hidden = out.hidden_states[-1][:, -1:, :]  # (1, 1, drafter_dim)
            all_hidden.append(cur_hidden)
            last_token = token

        return draft_tokens, draft_log_probs, drafter_past, all_hidden
