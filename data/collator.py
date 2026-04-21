"""
DataCollator: pads and batches multimodal samples.

Handles variable-length sequences and variable-size pixel_values.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


class DataCollator:
    """
    Collates samples from MultimodalDataset into a padded batch.

    Args:
        pad_token_id: token id used for padding input_ids.
        padding_side: "right" or "left" (HF convention).
    """

    def __init__(self, pad_token_id: int = 0, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, Any] = {}

        # ---- input_ids and labels ----
        input_ids_list = [s["input_ids"] for s in samples]
        labels_list = [s["labels"] for s in samples]
        max_len = max(x.shape[0] for x in input_ids_list)

        batch["input_ids"] = self._pad_sequences(
            input_ids_list, max_len, self.pad_token_id
        )
        batch["labels"] = self._pad_sequences(
            labels_list, max_len, IGNORE_INDEX
        )
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()

        # ---- pixel_values ----
        # Qwen2-VL processor returns pixel_values as (total_patches, C*t*ph*pw) —
        # there is NO batch dimension.  Always concatenate so the model receives a
        # single flat (total_patches_all_images, D) tensor; image_grid_thw tells
        # the visual encoder where each image's patches begin and end.
        pixel_values = [s["pixel_values"] for s in samples]
        if pixel_values[0] is not None:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)

        # ---- image_grid_thw ----
        grid_thw = [s.get("image_grid_thw") for s in samples]
        if grid_thw[0] is not None:
            batch["image_grid_thw"] = torch.stack(grid_thw, dim=0)

        return batch

    # ------------------------------------------------------------------ #

    def _pad_sequences(
        self,
        sequences: list[torch.LongTensor],
        max_len: int,
        pad_value: int,
    ) -> torch.LongTensor:
        padded = []
        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if self.padding_side == "right":
                padded.append(F.pad(seq, (0, pad_len), value=pad_value))
            else:
                padded.append(F.pad(seq, (pad_len, 0), value=pad_value))
        return torch.stack(padded, dim=0)
