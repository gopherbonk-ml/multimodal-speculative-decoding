"""
Classical Speculative Decoding for multimodal LLMs.

Algorithm (Leviathan et al., 2023 / Chen et al., 2023):
  1. Drafter generates γ tokens autoregressively (cheap).
  2. Target verifies all γ tokens in one parallel forward pass.
  3. Tokens are accepted/rejected using rejection sampling:
       accept token t_i if  U(0,1) < p_target(t_i) / p_draft(t_i)
     On first rejection: resample from adjusted distribution and stop.
  4. If all γ accepted: also emit target's greedy next token.
  => On average accepts β*γ + 1 tokens per target call (β = acceptance rate).

Multimodal considerations:
  - The image is processed ONCE and cached in the KV cache.
  - Subsequent draft/verify steps reuse the cached image KV (text-only appends).
  - The drafter uses the same image to build its own KV cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ..models.target import TargetModel
from ..models.base_drafter import BaseDrafter


@dataclass
class SpeculativeDecodingConfig:
    gamma: int = 5                     # number of draft tokens per step
    temperature: float = 1.0          # sampling temperature (1.0 = no scaling)
    top_p: float = 1.0                # nucleus sampling (1.0 = disabled)
    max_new_tokens: int = 256
    eos_token_id: int = 151645        # Qwen2 default <|endoftext|>
    pad_token_id: int = 151643
    do_sample: bool = True            # False = greedy (deterministic)
    # Stats collection
    collect_stats: bool = True


@dataclass
class DecodingStats:
    total_tokens_generated: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_target_calls: int = 0
    wall_time_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.wall_time_seconds == 0:
            return 0.0
        return self.total_tokens_generated / self.wall_time_seconds

    @property
    def mean_tokens_per_target_call(self) -> float:
        if self.total_target_calls == 0:
            return 0.0
        return self.total_tokens_generated / self.total_target_calls

    def __str__(self) -> str:
        return (
            f"Tokens generated: {self.total_tokens_generated} | "
            f"Acceptance rate: {self.acceptance_rate:.2%} | "
            f"Mean tokens/target call: {self.mean_tokens_per_target_call:.2f} | "
            f"Tokens/sec: {self.tokens_per_second:.1f}"
        )


class SpeculativeDecoder:
    """
    Wraps a (target, drafter) pair and implements speculative decoding.

    Both models must be on the same device and support KV-cache via
    `use_cache=True` and `past_key_values` arguments.
    """

    def __init__(
        self,
        target: TargetModel,
        drafter: BaseDrafter,
        config: SpeculativeDecodingConfig | None = None,
    ):
        self.target = target
        self.drafter = drafter
        self.cfg = config or SpeculativeDecodingConfig()

    # ------------------------------------------------------------------ #
    # Main entry point                                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, DecodingStats]:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids:      (1, prompt_len) — single batch only for now
            pixel_values:   image pixels for multimodal input
            image_grid_thw: Qwen2-VL grid info

        Returns:
            (output_ids, stats)  where output_ids: (1, prompt_len + new_tokens)
        """
        assert input_ids.shape[0] == 1, "Batch size must be 1 for speculative decoding"

        stats = DecodingStats()
        t_start = time.perf_counter()

        # ---- Prefill both target and drafter with the prompt + image ----
        target_past, drafter_past = self._prefill(
            input_ids, pixel_values, image_grid_thw, attention_mask
        )

        generated = input_ids.clone()  # (1, L)

        while generated.shape[1] - input_ids.shape[1] < self.cfg.max_new_tokens:
            if self._is_done(generated):
                break

            # ---- Draft phase ----
            draft_tokens, draft_log_probs, drafter_past = self._draft(
                generated, drafter_past
            )

            # ---- Verify phase ----
            accepted_tokens, bonus_token, target_past = self._verify(
                generated, draft_tokens, draft_log_probs, target_past
            )

            n_accepted = accepted_tokens.shape[1]
            stats.total_draft_tokens += len(draft_tokens)
            stats.total_accepted_tokens += n_accepted + 1  # +1 for bonus token
            stats.total_target_calls += 1

            # Append accepted tokens
            new_tokens = torch.cat([accepted_tokens, bonus_token], dim=1)  # (1, n_accepted + 1)
            generated = torch.cat([generated, new_tokens], dim=1)
            stats.total_tokens_generated += new_tokens.shape[1]

            # Roll drafter back to the correct position
            drafter_past = self._rollback_drafter_cache(
                drafter_past, n_accepted + 1  # + bonus
            )

        stats.wall_time_seconds = time.perf_counter() - t_start
        return generated, stats

    # ------------------------------------------------------------------ #
    # Prefill                                                             #
    # ------------------------------------------------------------------ #

    def _prefill(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ):
        """Run target and drafter on the prompt (including image) and return KV caches."""
        prefix = input_ids[:, :-1]

        # Target prefill
        target_out = self.target(
            input_ids=prefix,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask[:, :-1] if attention_mask is not None else None,
            use_cache=True,
        )
        target_past = target_out.past_key_values

        # Drafter prefill (Arch4 ignores pixel_values internally)
        drafter_out = self.drafter(
            input_ids=prefix,
            attention_mask=attention_mask[:, :-1] if attention_mask is not None else None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
        )
        drafter_past = drafter_out.past_key_values

        return target_past, drafter_past

    # ------------------------------------------------------------------ #
    # Draft phase                                                         #
    # ------------------------------------------------------------------ #

    def _draft(
        self,
        context: torch.LongTensor,
        drafter_past,
    ) -> tuple[list[torch.LongTensor], list[torch.Tensor], object]:
        """
        Auto-regressively generate γ tokens with the drafter.

        Returns:
            draft_tokens:    list of γ token tensors, each (1, 1)
            draft_log_probs: list of γ log-prob vectors over vocab, each (1, V)
            updated_past:    drafter KV cache after all γ steps
        """
        draft_tokens = []
        draft_log_probs = []

        # The drafter's KV cache already covers the entire context after prefill.
        # We just need to pass the last generated token at each step.
        last_token = context[:, -1:]  # (1, 1)

        for _ in range(self.cfg.gamma):
            out = self.drafter(
                input_ids=last_token,
                pixel_values=None,   # image already in KV cache
                image_grid_thw=None,
                use_cache=True,
                past_key_values=drafter_past,
            )
            logits = out.logits[:, -1, :]   # (1, V)
            log_probs = F.log_softmax(logits / max(self.cfg.temperature, 1e-6), dim=-1)

            token = self._sample(log_probs)  # (1, 1)
            draft_tokens.append(token)
            draft_log_probs.append(log_probs)

            drafter_past = out.past_key_values
            last_token = token

        return draft_tokens, draft_log_probs, drafter_past

    # ------------------------------------------------------------------ #
    # Verify phase                                                        #
    # ------------------------------------------------------------------ #

    def _verify(
        self,
        context: torch.LongTensor,
        draft_tokens: list[torch.LongTensor],
        draft_log_probs: list[torch.Tensor],
        target_past,
    ) -> tuple[torch.LongTensor, torch.LongTensor, object]:
        """
        Verify draft tokens with the target in a single forward pass.

        The target processes the γ draft tokens in parallel (appended to context).
        We then apply rejection sampling token by token.

        Returns:
            accepted_tokens: (1, n_accepted) — the accepted prefix
            bonus_token:     (1, 1)          — target's own next token prediction
            updated_target_past: target KV cache up to the accepted position
        """
        gamma = len(draft_tokens)

        # Build the sequence: [last context token] + draft_tokens
        # Target KV cache already covers context[:-1]; we feed context[-1] + drafts.
        draft_seq = torch.cat([context[:, -1:]] + draft_tokens, dim=1)  # (1, γ+1)

        target_out = self.target(
            input_ids=draft_seq,
            pixel_values=None,  # image in KV cache
            use_cache=True,
            past_key_values=target_past,
        )
        # Logits at positions 0..γ-1 correspond to predictions for draft tokens t_1..t_γ
        # Logit at position γ is the target's prediction for the (γ+1)-th token (bonus)
        target_logits = target_out.logits  # (1, γ+1, V)

        accepted = []
        n_accepted = 0
        # When rejection occurs, the resampled token is stored here and used directly
        # as bonus — no second sampling from target_logits at the same position.
        override_bonus: torch.LongTensor | None = None

        for i in range(gamma):
            t_logits = target_logits[:, i, :]    # (1, V) — target's dist at position i
            t_log_p = F.log_softmax(t_logits / max(self.cfg.temperature, 1e-6), dim=-1)
            d_log_p = draft_log_probs[i]         # (1, V)

            token = draft_tokens[i]              # (1, 1)
            token_idx = token.item()

            if self.cfg.do_sample:
                # Rejection sampling
                t_p = t_log_p[:, token_idx].exp()
                d_p = d_log_p[:, token_idx].exp()
                ratio = (t_p / (d_p + 1e-10)).clamp(max=1.0)

                u = torch.rand(1, device=token.device)
                if u <= ratio:
                    accepted.append(token)
                    n_accepted += 1
                else:
                    # Resample from adjusted distribution p_target - p_draft (clipped to 0).
                    # This resampled token IS the final output for this round — do not
                    # append it to `accepted` and do not sample an additional bonus token.
                    adjusted = (t_log_p.exp() - d_log_p.exp()).clamp(min=0.0)
                    s = adjusted.sum()
                    if s > 1e-10:
                        adjusted = adjusted / s
                        override_bonus = torch.multinomial(adjusted, 1)  # (1, 1)
                    else:
                        override_bonus = self._sample(t_log_p)
                    break
            else:
                # Greedy: accept if target also picks the same token
                target_pick = t_logits.argmax(-1, keepdim=True)  # (1, 1)
                if target_pick.item() == token_idx:
                    accepted.append(token)
                    n_accepted += 1
                else:
                    override_bonus = target_pick
                    break

        # Bonus token: resampled (on rejection) or target's greedy/sampled next token
        # (when all γ draft tokens were accepted).
        if override_bonus is not None:
            bonus_token = override_bonus
        else:
            bonus_logits = target_logits[:, gamma, :]
            bonus_log_p = F.log_softmax(bonus_logits / max(self.cfg.temperature, 1e-6), dim=-1)
            bonus_token = self._sample(bonus_log_p) if self.cfg.do_sample else bonus_logits.argmax(-1, keepdim=True)

        if accepted:
            accepted_ids = torch.cat(accepted, dim=1)  # (1, n_accepted)
        else:
            accepted_ids = torch.empty(1, 0, dtype=torch.long, device=context.device)

        # Trim target KV cache: keep [context[-1], draft_t[0], ..., draft_t[n_accepted-1]].
        # That is n_accepted+1 entries from the γ+1 tokens we just fed.
        trim_len = n_accepted + 1
        trimmed_past = self._trim_cache(target_out.past_key_values, trim_len, gamma + 1)

        return accepted_ids, bonus_token, trimmed_past

    # ------------------------------------------------------------------ #
    # KV cache manipulation                                               #
    # ------------------------------------------------------------------ #

    def _trim_cache(self, past_key_values, keep: int, total_new: int):
        """
        Trim the last (total_new - keep) entries from each layer's KV cache.
        Works with the standard list-of-tuples format from HuggingFace.
        """
        if keep == total_new:
            return past_key_values

        # Number of positions to drop from the end
        drop = total_new - keep
        trimmed = []
        for k, v in past_key_values:
            trimmed.append((k[:, :, :-drop, :], v[:, :, :-drop, :]))
        return tuple(trimmed)

    def _rollback_drafter_cache(self, drafter_past, n_tokens_to_keep_new: int):
        """
        After verification we need to roll the drafter cache back to the accepted position.
        The drafter ran γ steps forward; we keep only n_tokens_to_keep_new of them.
        """
        if drafter_past is None:
            return None
        drop = self.cfg.gamma - (n_tokens_to_keep_new - 1)  # -1 because bonus is from target
        if drop <= 0:
            return drafter_past
        trimmed = []
        for k, v in drafter_past:
            trimmed.append((k[:, :, :-drop, :], v[:, :, :-drop, :]))
        return tuple(trimmed)

    # ------------------------------------------------------------------ #
    # Sampling                                                            #
    # ------------------------------------------------------------------ #

    def _sample(self, log_probs: torch.Tensor) -> torch.LongTensor:
        """Sample from log probability distribution with optional top-p."""
        if not self.cfg.do_sample:
            return log_probs.argmax(-1, keepdim=True)

        probs = log_probs.exp()

        if self.cfg.top_p < 1.0:
            probs = self._top_p_filter(probs, self.cfg.top_p)

        token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        return token

    @staticmethod
    def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        # Scatter back to original order
        out = torch.zeros_like(probs)
        out.scatter_(-1, sorted_idx, sorted_probs)
        # Renormalise
        total = out.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return out / total

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _is_done(self, tokens: torch.LongTensor) -> bool:
        return (tokens[0, -1].item() == self.cfg.eos_token_id)
