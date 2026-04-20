"""
Entry point for distillation training.

Usage (single GPU):
    python train.py --config configs/distill_arch1.yaml

Usage (multi-GPU, 8×V100):
    torchrun --nproc_per_node=8 train.py --config configs/distill_arch1.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import yaml
import torch
from transformers import Qwen2VLProcessor

from models import TargetModel, Arch1Drafter, Arch2Drafter, Arch3Drafter, Arch4Drafter, Arch5Eagle3Drafter
from distillation import DistillationTrainer, Eagle3Trainer, Eagle3TrainingConfig
from distillation.trainer import TrainingConfig
from data import build_datasets, DataCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ARCH_MAP = {
    "arch1": Arch1Drafter,
    "arch2": Arch2Drafter,
    "arch3": Arch3Drafter,
    "arch4": Arch4Drafter,
    "eagle3": Arch5Eagle3Drafter,
}

# Architectures that require Eagle3Trainer instead of DistillationTrainer
EAGLE3_ARCHS = {"eagle3"}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_drafter(arch: str, target: TargetModel, arch_kwargs: dict):
    cls = ARCH_MAP[arch]
    return cls(target=target, **arch_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Target model ----
    logger.info(f"Loading target model from {cfg['target_model']}")
    target = TargetModel(
        model_name_or_path=cfg["target_model"],
        torch_dtype=torch.bfloat16,
    )

    # ---- Processor ----
    processor = Qwen2VLProcessor.from_pretrained(cfg["target_model"])

    # ---- Drafter ----
    arch = cfg["arch"]
    logger.info(f"Building drafter architecture: {arch}")
    drafter = build_drafter(arch, target, cfg.get("arch_kwargs", {}))
    n_params = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
    logger.info(f"Trainable drafter parameters: {n_params / 1e6:.1f}M")

    # ---- Datasets ----
    logger.info("Loading datasets...")
    train_dataset = build_datasets(
        source_config_path=cfg["data_config"],
        processor=processor,
        max_seq_len=cfg.get("max_seq_len", 2048),
        split="train",
    )
    eval_dataset = build_datasets(
        source_config_path=cfg["data_config"],
        processor=processor,
        max_seq_len=cfg.get("max_seq_len", 2048),
        split="eval",
    )
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    collator = DataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    # ---- Training config & trainer ----
    training_kwargs = cfg.get("training", {})
    if arch in EAGLE3_ARCHS:
        train_cfg = Eagle3TrainingConfig(**training_kwargs)
        trainer = Eagle3Trainer(
            target=target,
            drafter=drafter,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            config=train_cfg,
        )
    else:
        train_cfg = TrainingConfig(**training_kwargs)
        trainer = DistillationTrainer(
            target=target,
            drafter=drafter,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            config=train_cfg,
        )

    trainer.train()


if __name__ == "__main__":
    main()
