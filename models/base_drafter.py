"""
BaseDrafter: abstract base class for all drafter architectures.

Every drafter must implement:
  - forward()        — standard training forward pass
  - draft()          — generate one token given current KV cache (for inference)
  - prepare_vision() — process image and return visual token embeddings
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseDrafter(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        labels: torch.LongTensor | None,
        **kwargs,
    ):
        """Full forward pass returning a ModelOutput with at least `.logits`."""
        ...

    @abstractmethod
    def prepare_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process image and return visual embeddings.

        Returns:
            (total_image_tokens, drafter_hidden_size)
        """
        ...

    def build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        visual_embeds: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        """
        Replace IMAGE token positions in `input_ids` with `visual_embeds`.

        This mirrors the logic inside Qwen2VLForConditionalGeneration.forward().
        """
        text_embeds = self.embed_tokens(input_ids)  # (B, L, D)

        # Flatten to (B*L, D) for easier indexing
        B, L, D = text_embeds.shape
        flat = text_embeds.view(-1, D)
        image_mask = (input_ids == image_token_id).view(-1)

        n_image_tokens = image_mask.sum().item()
        if n_image_tokens != visual_embeds.shape[0]:
            raise ValueError(
                f"Number of image token positions ({n_image_tokens}) does not match "
                f"visual_embeds length ({visual_embeds.shape[0]})"
            )

        flat[image_mask] = visual_embeds.to(flat.dtype)
        return flat.view(B, L, D)

    @property
    @abstractmethod
    def embed_tokens(self) -> nn.Embedding:
        """Token embedding layer of the drafter LLM."""
        ...

    @property
    @abstractmethod
    def image_token_id(self) -> int:
        ...
