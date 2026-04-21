"""
EAGLE-3 training loss.

Combines three terms:
    1. Token distillation  — KL(target || drafter) in log-prob space
    2. Feature alignment   — MSE between drafter's last-layer hidden states and
                             target's hidden states projected to the same feature space.
                             Both sides are L2-normalised before MSE to decouple
                             scale from direction.
    3. Task loss           — cross-entropy with ground-truth labels (optional)

    total = alpha * distill  +  beta * feature  +  (1 - alpha - beta) * task
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.losses import DistillationLoss


class Eagle3Loss(nn.Module):
    """
    Args:
        temperature: KL softmax temperature.
        alpha:       weight for token-distillation term (0 < alpha ≤ 1).
        beta:        weight for feature-alignment term (0 ≤ beta, alpha+beta ≤ 1).
        loss_type:   distillation variant — see DistillationLoss.VALID_LOSS_TYPES.
        top_k:       for loss_type="topk_kl".
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,
        beta: float = 0.2,
        loss_type: str = "forward_kl",
        top_k: int = 50,
    ):
        super().__init__()
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0 <= beta):
            raise ValueError(f"beta must be >= 0, got {beta}")
        if alpha + beta > 1.0:
            raise ValueError(f"alpha + beta must be <= 1.0, got {alpha + beta}")

        self.alpha = alpha
        self.beta = beta
        self.task_weight = 1.0 - alpha - beta

        # Reuse DistillationLoss for the KL/JS/topk_kl computation.
        # alpha=1.0 here means it only computes distill_loss (task loss handled here).
        self._distill_fn = DistillationLoss(
            temperature=temperature,
            alpha=1.0,
            loss_type=loss_type,
            top_k=top_k,
        )

    def forward(
        self,
        drafter_logits: torch.Tensor,
        target_logits: torch.Tensor,
        drafter_hidden: torch.Tensor,
        target_features_projected: torch.Tensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            drafter_logits:             (B, L, V)
            target_logits:              (B, L, V)  — detached, no grad
            drafter_hidden:             (B, L, drafter_dim)  — last LLM layer
            target_features_projected:  (B, L, feature_dim)  — via project_target_features()
            labels:                     (B, L) with -100 at ignored positions
            attention_mask:             (B, L)
        Returns:
            dict with keys: loss, distill_loss, feature_loss, task_loss
        """
        # ---- 1. Token distillation ----
        distill_out = self._distill_fn(
            drafter_logits=drafter_logits,
            target_logits=target_logits,
            labels=labels,
            attention_mask=attention_mask,
        )
        distill_loss = distill_out["distill_loss"]

        # ---- 2. Feature alignment ----
        # Compute at all valid (non-padding) positions.
        if attention_mask is not None:
            feat_mask = attention_mask.bool()
        else:
            feat_mask = torch.ones(
                drafter_hidden.shape[:2], dtype=torch.bool, device=drafter_hidden.device
            )

        d_h = drafter_hidden[feat_mask]             # (N, drafter_dim)
        t_h = target_features_projected[feat_mask]  # (N, feature_dim)

        # L2-normalise both sides so the loss measures directional similarity only.
        d_h_norm = F.normalize(d_h.float(), dim=-1)
        t_h_norm = F.normalize(t_h.float(), dim=-1)
        feature_loss = F.mse_loss(d_h_norm, t_h_norm)

        # ---- 3. Task loss (CE) ----
        task_loss = torch.tensor(0.0, device=drafter_logits.device)
        if labels is not None and self.task_weight > 0:
            shifted_logits = drafter_logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                valid = attention_mask[:, 1:].bool() & (shifted_labels != -100)
            else:
                valid = shifted_labels != -100
            if valid.any():
                task_loss = F.cross_entropy(
                    shifted_logits[valid],
                    shifted_labels[valid],
                )

        # ---- Combine ----
        loss = (
            self.alpha * distill_loss
            + self.beta * feature_loss.to(distill_loss.dtype)
            + self.task_weight * task_loss
        )

        return {
            "loss": loss,
            "distill_loss": distill_loss.detach(),
            "feature_loss": feature_loss.detach().to(distill_loss.dtype),
            "task_loss": task_loss.detach(),
        }
