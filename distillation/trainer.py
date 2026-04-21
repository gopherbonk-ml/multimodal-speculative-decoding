"""
DistillationTrainer: training loop for distilling target → drafter.

Features:
  - Multi-GPU via PyTorch DDP (8×V100 setup)
  - Gradient checkpointing support
  - Mixed precision (bf16 / fp16)
  - Periodic evaluation and checkpoint saving
  - Logging to console (and optionally W&B)
"""

from __future__ import annotations

import os
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from distillation.losses import DistillationLoss
from models.base_drafter import BaseDrafter
from models.target import TargetModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Optimiser
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03

    # Batch
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Checkpointing
    output_dir: str = "checkpoints"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10

    # Distillation
    temperature: float = 2.0
    alpha: float = 0.9
    loss_type: str = "forward_kl"
    top_k: int = 50

    # Misc
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "multimodal-speculative-decoding"
    dataloader_num_workers: int = 4


class DistillationTrainer:
    """
    Trains a drafter model to mimic the target via knowledge distillation.

    Usage:
        trainer = DistillationTrainer(target, drafter, train_dataset, eval_dataset, config)
        trainer.train()
    """

    def __init__(
        self,
        target: TargetModel,
        drafter: BaseDrafter,
        train_dataset,
        eval_dataset,
        data_collator,
        config: TrainingConfig,
    ):
        self.config = config
        self.target = target
        self.drafter = drafter
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        self._setup_distributed()
        self._setup_dtype()
        self._setup_models()
        self._setup_loss()
        self._setup_optimizer_and_scheduler()
        self._setup_logging()

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #

    def _setup_distributed(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.local_rank == 0

        if self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

    def _setup_dtype(self):
        if self.config.bf16:
            self.dtype = torch.bfloat16
            self.use_amp = True
        elif self.config.fp16:
            self.dtype = torch.float16
            self.use_amp = True
        else:
            self.dtype = torch.float32
            self.use_amp = False

        self.scaler = GradScaler() if (self.use_amp and self.config.fp16) else None

    def _setup_models(self):
        self.target = self.target.to(self.device)
        self.target.eval()

        self.drafter = self.drafter.to(self.device)
        self.drafter.train()

        if self.world_size > 1:
            self.drafter = DDP(
                self.drafter,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

    def _setup_loss(self):
        self.loss_fn = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            loss_type=self.config.loss_type,
            top_k=self.config.top_k,
        )

    def _setup_optimizer_and_scheduler(self):
        drafter_module = self.drafter.module if self.world_size > 1 else self.drafter
        trainable_params = [p for p in drafter_module.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler is set up after we know total steps
        self._total_steps = None
        self.scheduler = None

    def _setup_logging(self):
        if self.is_main and self.config.use_wandb:
            try:
                import wandb
                wandb.init(project=self.config.wandb_project)
            except ImportError:
                logger.warning("wandb not installed; skipping W&B logging.")

    # ------------------------------------------------------------------ #
    # DataLoaders                                                         #
    # ------------------------------------------------------------------ #

    def _make_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self.world_size > 1 else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self):
        train_loader = self._make_dataloader(
            self.train_dataset,
            self.config.per_device_train_batch_size,
            shuffle=True,
        )

        steps_per_epoch = math.ceil(len(train_loader) / self.config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        if self.is_main:
            logger.info(f"Training for {total_steps} optimizer steps ({self.config.num_train_epochs} epochs)")

        global_step = 0
        self.optimizer.zero_grad()

        for epoch in range(self.config.num_train_epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss_dict = self._training_step(batch)
                loss = loss_dict["loss"] / self.config.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    nn.utils.clip_grad_norm_(
                        [p for p in self.drafter.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if self.is_main and global_step % self.config.logging_steps == 0:
                        self._log(global_step, loss_dict)

                    if self.is_main and global_step % self.config.eval_steps == 0:
                        self.evaluate()

                    if self.is_main and global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step)

        if self.is_main:
            self._save_checkpoint(global_step, final=True)

        if self.world_size > 1:
            dist.destroy_process_group()

    @torch.no_grad()
    def _get_target_logits(self, batch: dict) -> torch.Tensor:
        """Run target model in inference mode."""
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            out = self.target(
                input_ids=batch.get("input_ids"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch.get("attention_mask"),
                use_cache=False,
            )
        return out.logits  # (B, L, V)

    def _training_step(self, batch: dict) -> dict[str, torch.Tensor]:
        target_logits = self._get_target_logits(batch)

        with autocast(enabled=self.use_amp, dtype=self.dtype):
            drafter_out = self.drafter(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                labels=batch.get("labels"),
                use_cache=False,
            )

        drafter_logits = drafter_out.logits  # (B, L, V)

        loss_dict = self.loss_fn(
            drafter_logits=drafter_logits,
            target_logits=target_logits.detach(),
            labels=batch.get("labels"),
            attention_mask=batch.get("attention_mask"),
        )

        return loss_dict

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self) -> dict[str, float]:
        eval_loader = self._make_dataloader(
            self.eval_dataset,
            self.config.per_device_eval_batch_size,
            shuffle=False,
        )

        drafter_module = self.drafter.module if self.world_size > 1 else self.drafter
        drafter_module.eval()

        total_loss = 0.0
        total_distill = 0.0
        total_task = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                loss_dict = self._training_step(batch)
                total_loss += loss_dict["loss"].item()
                total_distill += loss_dict["distill_loss"].item()
                total_task += loss_dict["task_loss"].item()
                n_batches += 1

        drafter_module.train()

        metrics = {
            "eval/loss": total_loss / max(n_batches, 1),
            "eval/distill_loss": total_distill / max(n_batches, 1),
            "eval/task_loss": total_task / max(n_batches, 1),
        }

        if self.is_main:
            logger.info(f"Eval: {metrics}")
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.log(metrics)
                except ImportError:
                    pass

        return metrics

    # ------------------------------------------------------------------ #
    # Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _log(self, step: int, loss_dict: dict):
        metrics = {
            "train/loss": loss_dict["loss"].item(),
            "train/distill_loss": loss_dict["distill_loss"].item(),
            "train/task_loss": loss_dict["task_loss"].item(),
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/step": step,
        }
        logger.info(f"Step {step}: {metrics}")
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass

    def _save_checkpoint(self, step: int, final: bool = False):
        os.makedirs(self.config.output_dir, exist_ok=True)
        tag = "final" if final else f"step-{step}"
        save_path = os.path.join(self.config.output_dir, tag)
        os.makedirs(save_path, exist_ok=True)

        drafter_module = self.drafter.module if self.world_size > 1 else self.drafter
        # Save drafter weights only (target is frozen and not ours to save)
        torch.save(drafter_module.state_dict(), os.path.join(save_path, "drafter.pt"))
        logger.info(f"Saved checkpoint to {save_path}")
