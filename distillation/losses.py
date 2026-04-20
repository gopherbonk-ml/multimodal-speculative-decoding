"""
Distillation losses for training the drafter to mimic the target.

Supported losses:
  - KL divergence on output distributions (forward KL)
  - Reverse KL
  - Jensen-Shannon divergence
  - Top-k KL (only over top-k target tokens, more stable)
  - Cross-entropy with soft labels
  - Combined: task LM loss + distillation loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Computes a weighted combination of:
      - Soft distillation loss (KL / JS / top-k KL) between drafter and target logits
      - Hard task loss (cross-entropy with ground-truth labels)

    Args:
        temperature:   Softmax temperature for soft labels. Higher = softer.
        alpha:         Weight for the distillation term (1-alpha for task loss).
        loss_type:     One of {"forward_kl", "reverse_kl", "js", "topk_kl", "soft_ce"}.
        top_k:         Used only when loss_type="topk_kl".
        reduction:     "mean" or "sum".
    """

    VALID_LOSS_TYPES = {"forward_kl", "reverse_kl", "js", "topk_kl", "soft_ce"}

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.9,
        loss_type: str = "forward_kl",
        top_k: int = 50,
        reduction: str = "mean",
    ):
        super().__init__()
        if loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {self.VALID_LOSS_TYPES}")
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type
        self.top_k = top_k
        self.reduction = reduction

    def forward(
        self,
        drafter_logits: torch.Tensor,
        target_logits: torch.Tensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            drafter_logits: (B, L, V) — drafter output logits
            target_logits:  (B, L, V) — target output logits (no grad)
            labels:         (B, L)    — ground-truth token ids; -100 = ignore
            attention_mask: (B, L)    — 1 for valid positions

        Returns:
            dict with keys: "loss", "distill_loss", "task_loss"
        """
        # Shift for next-token prediction
        drafter_logits = drafter_logits[:, :-1, :].contiguous()
        target_logits = target_logits[:, :-1, :].contiguous()

        if attention_mask is not None:
            mask = attention_mask[:, 1:].bool()  # shifted
        else:
            mask = torch.ones(drafter_logits.shape[:2], dtype=torch.bool, device=drafter_logits.device)

        if labels is not None:
            shifted_labels = labels[:, 1:].contiguous()
            valid_mask = mask & (shifted_labels != -100)
        else:
            valid_mask = mask

        # Flatten to (N_valid, V)
        drafter_flat = drafter_logits[valid_mask]
        target_flat = target_logits[valid_mask]

        distill_loss = self._distill_loss(drafter_flat, target_flat)

        task_loss = torch.tensor(0.0, device=drafter_logits.device)
        if labels is not None and self.alpha < 1.0:
            task_loss = F.cross_entropy(
                drafter_flat,
                shifted_labels[valid_mask],
                reduction=self.reduction,
            )

        loss = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss

        return {
            "loss": loss,
            "distill_loss": distill_loss.detach(),
            "task_loss": task_loss.detach(),
        }

    def _distill_loss(self, drafter_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        drafter_log_prob = F.log_softmax(drafter_logits / T, dim=-1)
        target_prob = F.softmax(target_logits / T, dim=-1)

        if self.loss_type == "forward_kl":
            # KL(target || drafter): minimise when drafter covers all target mass
            loss = F.kl_div(drafter_log_prob, target_prob, reduction="batchmean") * (T ** 2)

        elif self.loss_type == "reverse_kl":
            # KL(drafter || target): mode-seeking
            drafter_prob = drafter_log_prob.exp()
            target_log_prob = F.log_softmax(target_logits / T, dim=-1)
            loss = F.kl_div(target_log_prob, drafter_prob, reduction="batchmean") * (T ** 2)

        elif self.loss_type == "js":
            drafter_prob = drafter_log_prob.exp()
            m = 0.5 * (drafter_prob + target_prob)
            log_m = m.log().clamp(min=-100)
            kl_d = F.kl_div(log_m, drafter_prob, reduction="batchmean")
            kl_t = F.kl_div(log_m, target_prob, reduction="batchmean")
            loss = 0.5 * (kl_d + kl_t) * (T ** 2)

        elif self.loss_type == "topk_kl":
            # Only compute KL over the top-k tokens of the target distribution
            # More stable and focuses learning on likely tokens
            topk_vals, topk_idx = target_prob.topk(self.top_k, dim=-1)  # (N, k)
            # Renormalise the top-k target distribution
            topk_target = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            # Gather drafter log-probs at those positions
            topk_drafter_log = drafter_log_prob.gather(1, topk_idx)
            topk_drafter_log = topk_drafter_log - torch.logsumexp(topk_drafter_log, dim=-1, keepdim=True)
            loss = F.kl_div(topk_drafter_log, topk_target, reduction="batchmean") * (T ** 2)

        elif self.loss_type == "soft_ce":
            # Cross-entropy with soft target labels
            loss = -(target_prob * drafter_log_prob).sum(dim=-1)
            loss = loss.mean() * (T ** 2)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss
