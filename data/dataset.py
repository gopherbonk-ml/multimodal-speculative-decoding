"""
Dataset loading for multimodal distillation training.

Data format (per-sample JSONL):
    {
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "Describe the image."},
            {"from": "gpt",   "value": "The image shows ..."}
        ]
    }

Data source config (JSONL, one source per line):
    {
        "image_folder": "/data/coco/images",
        "jsonl_path":   "/data/coco/annotations.jsonl",
        "weight":       1.0,         # optional sampling weight
        "split":        "train"      # optional label for logging
    }
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from transformers import Qwen2VLProcessor


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


class MultimodalDataset(Dataset):
    """
    Single-source multimodal conversation dataset.

    Builds input_ids and labels from conversations, inserts image tokens,
    and loads images for the processor.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_folder: str,
        processor: Qwen2VLProcessor,
        max_seq_len: int = 2048,
        image_token: str = DEFAULT_IMAGE_TOKEN,
        split: str = "train",
        metadata: dict | None = None,  # extra info stored for logging
    ):
        self.image_folder = Path(image_folder)
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.image_token = image_token
        self.split = split
        self.metadata = metadata or {}

        self.samples: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.image_folder / sample["image"]
        conversations = sample["conversations"]

        # Build prompt text following Qwen2-VL chat template
        messages = self._build_messages(conversations, has_image=True)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback: blank image (avoids crashing on corrupted files)
            image = Image.new("RGB", (336, 336), color=0)

        # Tokenise via processor (handles image resizing, patch extraction, chat template)
        processed = self.processor(
            text=[self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = processed["input_ids"].squeeze(0)           # (L,)
        pixel_values = processed["pixel_values"].squeeze(0)     # depends on model
        image_grid_thw = processed.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)

        # Build labels: mask the human turns (only supervise assistant responses)
        labels = self._build_labels(input_ids, conversations)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        conversations: list[dict],
        has_image: bool,
    ) -> list[dict]:
        """
        Convert conversations list to Qwen2-VL messages format.

        apply_chat_template requires content to be a list of typed dicts
        ({"type": "image"} / {"type": "text", "text": ...}) for multimodal
        messages.  A plain string content will NOT trigger image-token insertion.
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"]

            if i == 0 and has_image and role == "user":
                # Image must come BEFORE the text in the content list
                content = [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ]
            else:
                content = [{"type": "text", "text": text}]

            messages.append({"role": role, "content": content})
        return messages

    def _build_labels(
        self,
        input_ids: torch.LongTensor,
        conversations: list[dict],
    ) -> torch.LongTensor:
        """
        Mask everything except assistant responses with IGNORE_INDEX.

        Scans for all <|im_start|>assistant\\n ... <|im_end|> blocks and
        supervises only the content between them.  Handles multi-turn correctly.
        """
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        header_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        end_ids = self.processor.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )

        pos = 0
        while pos < len(input_ids):
            header_pos = self._find_subsequence(input_ids[pos:], header_ids)
            if header_pos < 0:
                break
            content_start = pos + header_pos + len(header_ids)
            end_pos = self._find_subsequence(input_ids[content_start:], end_ids)
            if end_pos < 0:
                # No closing tag — supervise to end of sequence
                labels[content_start:] = input_ids[content_start:]
                break
            content_end = content_start + end_pos
            labels[content_start:content_end] = input_ids[content_start:content_end]
            pos = content_end + len(end_ids)

        return labels

    @staticmethod
    def _find_subsequence(seq: torch.LongTensor, sub: list[int]) -> int:
        """Return start index of first occurrence of `sub` in `seq`, or -1."""
        n, m = len(seq), len(sub)
        sub_t = torch.tensor(sub, dtype=seq.dtype, device=seq.device)
        for i in range(n - m + 1):
            if (seq[i: i + m] == sub_t).all():
                return i
        return -1


def build_datasets(
    source_config_path: str,
    processor: Qwen2VLProcessor,
    max_seq_len: int = 2048,
    split: str = "train",
    seed: int = 42,
) -> ConcatDataset:
    """
    Build a ConcatDataset from a multi-source config JSONL file.

    Each line in source_config_path:
        {
            "image_folder": "...",
            "jsonl_path":   "...",
            "weight":       1.0,   # optional, controls over/under-sampling
            "split":        "train"
        }
    """
    sources: list[dict] = []
    with open(source_config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sources.append(json.loads(line))

    datasets = []
    for src in sources:
        src_split = src.get("split", "train")
        if src_split != split:
            continue

        ds = MultimodalDataset(
            jsonl_path=src["jsonl_path"],
            image_folder=src["image_folder"],
            processor=processor,
            max_seq_len=max_seq_len,
            split=split,
            metadata={k: v for k, v in src.items() if k not in ("jsonl_path", "image_folder")},
        )

        # Repeat dataset according to weight (integer approximation)
        weight = src.get("weight", 1.0)
        repeat = max(1, round(weight))
        datasets.extend([ds] * repeat)

    if not datasets:
        raise ValueError(f"No datasets found for split='{split}' in {source_config_path}")

    return ConcatDataset(datasets)
