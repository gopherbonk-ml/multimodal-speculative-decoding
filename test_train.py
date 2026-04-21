"""
Minimal end-to-end training test.

Uses MockTarget (no weights downloaded) + synthetic data.
Runs DistillationTrainer and Eagle3Trainer for a handful of steps on CPU/MPS.

Usage:
    python test_train.py              # arch1 (default)
    python test_train.py --arch eagle3
    python test_train.py --arch all   # runs every arch
    python test_train.py --steps 5    # override number of optimizer steps
"""

from __future__ import annotations

import argparse
import sys
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_train")

# ─────────────────────────────────────────────────────────────────────────────
# Tiny dimensions
# ─────────────────────────────────────────────────────────────────────────────
T_HIDDEN = 64
T_VIT    = 48
VOCAB    = 512
SEQ      = 12
IMG_TOK  = 4
IMG_TOK_ID = 200
EOS_ID   = 1


# ─────────────────────────────────────────────────────────────────────────────
# MockTarget  (same as smoke_test.py)
# ─────────────────────────────────────────────────────────────────────────────

class _MockNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(T_VIT)
    def forward(self, x, **kw):
        return x


class MockVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_MockNorm()])
        self._merger = nn.Linear(T_VIT, T_HIDDEN, bias=False)

    def forward(self, pixel_values: torch.Tensor, grid_thw=None) -> torch.Tensor:
        B = pixel_values.shape[0] if pixel_values.ndim > 1 else 1
        raw = torch.zeros(IMG_TOK * B, T_VIT,
                          device=pixel_values.device, dtype=pixel_values.dtype)
        return self._merger(raw)

    def patch_embed(self, pixel_values):
        B = pixel_values.shape[0] if pixel_values.ndim > 1 else 1
        return torch.zeros(IMG_TOK * B, T_VIT,
                           device=pixel_values.device, dtype=pixel_values.dtype)

    def rot_pos_emb(self, grid_thw):
        return None


class MockTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self._visual = MockVisual()
        llm_cfg = Qwen2Config(
            vocab_size=VOCAB, hidden_size=T_HIDDEN, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1,
            intermediate_size=T_HIDDEN * 2, max_position_embeddings=512,
            rope_theta=10_000.0, tie_word_embeddings=False, use_sliding_window=False,
        )
        self._llm = Qwen2ForCausalLM(llm_cfg)

    @property
    def visual(self):          return self._visual
    @property
    def embed_tokens(self):    return self._llm.model.embed_tokens
    @property
    def lm_hidden_size(self):  return T_HIDDEN
    @property
    def vocab_size(self):      return VOCAB
    @property
    def image_token_id(self):  return IMG_TOK_ID
    @property
    def vit_hidden_size(self): return T_VIT

    @torch.no_grad()
    def get_raw_vit_features(self, pixel_values, image_grid_thw=None):
        B = pixel_values.shape[0] if pixel_values.ndim > 1 else 1
        return torch.zeros(IMG_TOK * B, T_VIT,
                           device=pixel_values.device, dtype=pixel_values.dtype)

    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None,
                attention_mask=None, past_key_values=None, inputs_embeds=None,
                use_cache=False, output_hidden_states=False, **kwargs):
        return self._llm(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_hidden_states=output_hidden_states,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """Generates random token sequences that mimic multimodal batches."""

    def __init__(self, size: int = 16, img_hw: int = 16):
        self.size = size
        self.img_hw = img_hw

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Stay below IMG_TOK_ID to avoid accidental image-token collisions
        ids = torch.randint(2, IMG_TOK_ID, (SEQ,))
        ids[:IMG_TOK] = IMG_TOK_ID
        mask = torch.ones(SEQ, dtype=torch.long)
        labels = ids.clone()
        labels[:IMG_TOK] = -100
        pixel_values = torch.rand(3, self.img_hw, self.img_hw)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Tiny drafter config
# ─────────────────────────────────────────────────────────────────────────────

def tiny_drafter_config():
    return Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
        max_position_embeddings=512, rope_theta=10_000.0,
        tie_word_embeddings=False, use_sliding_window=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patched trainer: adds MPS support + max_steps guard
# ─────────────────────────────────────────────────────────────────────────────

def _patch_trainer_for_local(trainer_cls, max_steps: int):
    """Monkey-patch device detection and add early-stop after max_steps."""

    original_setup_distributed = trainer_cls._setup_distributed
    original_train = trainer_cls.train

    def patched_setup_distributed(self):
        self.local_rank = 0
        self.world_size = 1
        self.is_main = True
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

    def patched_train(self):
        """Same as original but stops after max_steps optimizer steps."""
        import math
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
        import torch.nn as nn

        train_loader = self._make_dataloader(
            self.train_dataset, self.config.per_device_train_batch_size, shuffle=True
        )
        steps_per_epoch = math.ceil(len(train_loader) / self.config.gradient_accumulation_steps)
        total_steps = min(steps_per_epoch * self.config.num_train_epochs, max_steps)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        logger.info(f"Running up to {total_steps} optimizer steps")
        global_step = 0
        self.optimizer.zero_grad()

        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss_dict = self._training_step(batch)
                loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in self.drafter.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    self._log(global_step, loss_dict)

                    if global_step % self.config.eval_steps == 0:
                        self.evaluate()

                    if global_step >= max_steps:
                        logger.info(f"Reached max_steps={max_steps}, stopping.")
                        return

        logger.info("Training complete.")

    trainer_cls._setup_distributed = patched_setup_distributed
    trainer_cls.train = patched_train


def _patch_dataloader_no_pin(trainer_cls):
    """Disable pin_memory in DataLoaders (needed for MPS compatibility)."""
    from torch.utils.data import DataLoader, DistributedSampler

    def patched_make_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=self.data_collator, num_workers=0, pin_memory=False,
        )

    trainer_cls._make_dataloader = patched_make_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Per-arch training runs
# ─────────────────────────────────────────────────────────────────────────────

def run_arch(arch: str, target: MockTarget, max_steps: int):
    from distillation.trainer import DistillationTrainer, TrainingConfig

    cfg = TrainingConfig(
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        bf16=False, fp16=False,         # plain float32 — works on CPU/MPS without AMP
        output_dir=f"/tmp/test_train_{arch}",
        save_steps=9999,
        eval_steps=max_steps,
        logging_steps=1,
        warmup_ratio=0.1,
        use_wandb=False,
        dataloader_num_workers=0,
    )

    tiny = tiny_drafter_config()

    if arch == "arch1":
        from models.drafters.arch1 import Arch1Drafter
        drafter = Arch1Drafter(target=target, drafter_config=tiny)
    elif arch == "arch2":
        from models.drafters.arch2 import Arch2Drafter
        drafter = Arch2Drafter(target=target, drafter_config=tiny)
    elif arch == "arch3":
        from models.drafters.arch3 import Arch3Drafter
        drafter = Arch3Drafter(target=target, drafter_config=tiny,
                               img_size=28, patch_size=14,
                               vit_embed_dim=32, vit_depth=2, vit_num_heads=2)
    elif arch == "arch4":
        from models.drafters.arch4 import Arch4Drafter
        drafter = Arch4Drafter(target=target, drafter_config=tiny)
    elif arch == "eagle3":
        from models.drafters.arch5_eagle3 import Arch5Eagle3Drafter
        from distillation.eagle3_trainer import Eagle3Trainer, Eagle3TrainingConfig

        eagle_cfg = Eagle3TrainingConfig(
            **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__
               if k not in ("alpha", "beta")},
            alpha=0.7, beta=0.2,
        )
        drafter = Arch5Eagle3Drafter(target=target, drafter_config=tiny)
        train_ds = SyntheticDataset(size=16, img_hw=16)
        eval_ds  = SyntheticDataset(size=4,  img_hw=16)

        _patch_trainer_for_local(Eagle3Trainer, max_steps)
        _patch_dataloader_no_pin(Eagle3Trainer)

        trainer = Eagle3Trainer(
            target=target, drafter=drafter,
            train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collate, config=eagle_cfg,
        )
        logger.info(f"\n{'='*55}\nTraining Eagle3 ({arch})\n{'='*55}")
        trainer.train()
        return
    else:
        raise ValueError(f"Unknown arch: {arch}")

    img_hw = 28 if arch == "arch3" else 16
    train_ds = SyntheticDataset(size=16, img_hw=img_hw)
    eval_ds  = SyntheticDataset(size=4,  img_hw=img_hw)

    _patch_trainer_for_local(DistillationTrainer, max_steps)
    _patch_dataloader_no_pin(DistillationTrainer)

    trainer = DistillationTrainer(
        target=target, drafter=drafter,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate, config=cfg,
    )
    logger.info(f"\n{'='*55}\nTraining {arch}\n{'='*55}")
    trainer.train()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

ALL_ARCHS = ["arch1", "arch2", "arch3", "arch4", "eagle3"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="arch1",
                        help="arch1 | arch2 | arch3 | arch4 | eagle3 | all")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of optimizer steps per arch")
    args = parser.parse_args()

    archs = ALL_ARCHS if args.arch == "all" else [args.arch]

    logger.info("Building MockTarget...")
    target = MockTarget()

    for arch in archs:
        # Each arch gets a fresh copy of the patched trainer classes.
        # Re-importing forces a fresh class object so patches don't stack.
        import importlib
        import distillation.trainer as _dt
        import distillation.eagle3_trainer as _et
        importlib.reload(_dt)
        importlib.reload(_et)

        try:
            run_arch(arch, target, args.steps)
            logger.info(f"[OK] {arch} finished {args.steps} steps\n")
        except Exception as e:
            logger.error(f"[FAIL] {arch}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
