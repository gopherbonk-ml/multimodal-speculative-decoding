"""
Architecture 4: LLM(Drafter) only — text-only drafter.

No vision component. The image is completely ignored.
Serves as the lower-bound baseline:
"How well can a pure text drafter work on multimodal inputs?"

During inference the image tokens in the prompt are simply treated as
regular tokens (the image_token_id is present in input_ids, but no
actual image embeddings are injected — the model must rely on context).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.base_drafter import BaseDrafter
from models.target import TargetModel
from .small_llm_config import get_small_qwen2_config


class Arch4Drafter(BaseDrafter):
    """
    Text-only drafter: a small Qwen2 LM with no vision processing.
    """

    def __init__(
        self,
        target: TargetModel,
        drafter_config: Qwen2Config | None = None,
    ):
        super().__init__()

        self._image_token_id = target.image_token_id

        if drafter_config is None:
            drafter_config = get_small_qwen2_config(target.vocab_size)

        self.llm = Qwen2ForCausalLM(drafter_config)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.llm.model.embed_tokens

    @property
    def image_token_id(self) -> int:
        return self._image_token_id

    def prepare_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """No-op: text-only drafter ignores images."""
        raise RuntimeError(
            "Arch4Drafter has no vision component. "
            "Call forward() with pixel_values=None."
        )

    def build_inputs_embeds(self, input_ids, visual_embeds, image_token_id):
        """Arch4 never injects visual embeddings — just embed the tokens as-is."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,  # ignored
        image_grid_thw: torch.Tensor | None = None,  # ignored
        labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Silently ignore any image inputs
        return self.llm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
