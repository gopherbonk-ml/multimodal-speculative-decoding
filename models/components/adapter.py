"""
VisualAdapter: maps target-LLM-dim visual features → drafter-LLM-dim.

Used in Arch1: ViT(T) + Projector(T) → Adapter(D) → LLM(D).
The target visual module already outputs features in the target LLM hidden dim,
so the adapter bridges the two model sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualAdapter(nn.Module):
    """
    Lightweight linear adapter with optional gating.

    Maps (total_tokens, target_dim) → (total_tokens, drafter_dim).
    """

    def __init__(
        self,
        target_dim: int,
        drafter_dim: int,
        use_gate: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(target_dim, drafter_dim, bias=True)
        self.norm = nn.LayerNorm(drafter_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable gate initialised to 1 so that training starts from full pass-through
        if use_gate:
            self.gate = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("gate", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (total_tokens, target_dim)

        Returns:
            (total_tokens, drafter_dim)
        """
        out = self.proj(x)
        out = self.norm(out)
        out = self.dropout(out)
        if self.gate is not None:
            out = out * torch.sigmoid(self.gate)
        return out
