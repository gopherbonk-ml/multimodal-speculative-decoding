"""
Eagle3Trainer: extends DistillationTrainer for EAGLE-3 feature-conditioned training.

Key differences vs DistillationTrainer:
  1. Target runs with output_hidden_states=True to get last-layer hidden states.
  2. Hidden states are shifted by one position and projected via
     drafter.project_target_features() before being passed to the drafter.
  3. Loss is Eagle3Loss (token distillation + feature alignment + CE).
  4. Logging includes feature_loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch.cuda.amp import autocast

from .trainer import DistillationTrainer, TrainingConfig
from .eagle3_losses import Eagle3Loss
from ..models.drafters.arch5_eagle3 import Arch5Eagle3Drafter

logger = logging.getLogger(__name__)


@dataclass
class Eagle3TrainingConfig(TrainingConfig):
    """Extends TrainingConfig with EAGLE-3-specific hyper-parameters."""
    # Feature alignment loss weight
    beta: float = 0.2
    # Override default alpha to leave room for beta
    alpha: float = 0.7


class Eagle3Trainer(DistillationTrainer):
    """
    Training loop for Arch5Eagle3Drafter.

    Inherits all DDP / AMP / checkpointing / scheduling logic from
    DistillationTrainer and overrides only the forward pass and logging.
    """

    def __init__(self, target, drafter: Arch5Eagle3Drafter, *args, config: Eagle3TrainingConfig, **kwargs):
        # Pass config upward; DistillationTrainer stores it and builds optimiser / scheduler.
        super().__init__(target, drafter, *args, config=config, **kwargs)

        # Replace the generic DistillationLoss with Eagle3Loss.
        self.eagle3_loss_fn = Eagle3Loss(
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta,
            loss_type=config.loss_type,
            top_k=config.top_k,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _drafter_module(self) -> Arch5Eagle3Drafter:
        """Unwrap DDP to access Arch5Eagle3Drafter methods."""
        return self.drafter.module if self.world_size > 1 else self.drafter

    @torch.no_grad()
    def _get_target_outputs(self, batch: dict):
        """
        Run target in eval mode and return (logits, last_hidden_states).

        last_hidden_states: (B, L, target_hidden_dim)
        """
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            out = self.target(
                input_ids=batch.get("input_ids"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch.get("attention_mask"),
                use_cache=False,
                output_hidden_states=True,
            )
        # hidden_states is a tuple of (num_layers+1) tensors; take the last transformer layer.
        return out.logits, out.hidden_states[-1]

    def _build_shifted_features(
        self,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shift target hidden states right by one position so that
        h_{t-1} conditions the drafter input at position t.

        Position 0 receives an all-zeros feature vector.

        Args:
            target_hidden: (B, L, target_dim)
        Returns:
            projected_shifted: (B, L, feature_dim)
        """
        drafter = self._drafter_module()
        B, L, _ = target_hidden.shape
        device, dtype = target_hidden.device, target_hidden.dtype

        # Shift: [zeros, h_0, h_1, ..., h_{L-2}]
        h_shifted = torch.cat([
            torch.zeros(B, 1, target_hidden.shape[-1], device=device, dtype=dtype),
            target_hidden[:, :-1, :],
        ], dim=1)  # (B, L, target_dim)

        return drafter.project_target_features(h_shifted)  # (B, L, feature_dim)

    # ------------------------------------------------------------------ #
    # Training step                                                       #
    # ------------------------------------------------------------------ #

    def _training_step(self, batch: dict) -> dict[str, torch.Tensor]:
        # 1. Target forward (frozen, no grad)
        target_logits, target_hidden = self._get_target_outputs(batch)

        # 2. Build projected & shifted features for drafter input
        #    (detach so no gradient flows into the projection through target_hidden)
        projected_features = self._build_shifted_features(target_hidden.detach())

        # 3. Also project target hidden (non-shifted) for the alignment loss target
        drafter = self._drafter_module()
        target_features_for_align = drafter.project_target_features(
            target_hidden.detach()
        )  # (B, L, feature_dim)

        # 4. Drafter forward — always with output_hidden_states=True
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            drafter_out = self.drafter(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                projected_features=projected_features,
                use_cache=False,
                output_hidden_states=True,
            )

        drafter_logits = drafter_out.logits                  # (B, L, V)
        drafter_hidden = drafter_out.hidden_states[-1]       # (B, L, drafter_dim)

        # 5. Eagle3 loss
        loss_dict = self.eagle3_loss_fn(
            drafter_logits=drafter_logits,
            target_logits=target_logits.detach(),
            drafter_hidden=drafter_hidden,
            target_features_projected=target_features_for_align,
            labels=batch.get("labels"),
            attention_mask=batch.get("attention_mask"),
        )

        return loss_dict

    # ------------------------------------------------------------------ #
    # Logging (adds feature_loss)                                         #
    # ------------------------------------------------------------------ #

    def _log(self, step: int, loss_dict: dict):
        metrics = {
            "train/loss":         loss_dict["loss"].item(),
            "train/distill_loss": loss_dict["distill_loss"].item(),
            "train/feature_loss": loss_dict.get("feature_loss", torch.tensor(0.0)).item(),
            "train/task_loss":    loss_dict["task_loss"].item(),
            "train/lr":           self.scheduler.get_last_lr()[0],
            "train/step":         step,
        }
        logger.info(f"Step {step}: {metrics}")
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass

    def evaluate(self) -> dict[str, float]:
        metrics = super().evaluate()
        return metrics
