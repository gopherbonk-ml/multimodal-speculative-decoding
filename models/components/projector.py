"""
MLP projector: maps visual features → LLM embedding space.

Used in Arch2 (drafter projector with target ViT) and
Arch3 (drafter projector with drafter ViT).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Two-layer MLP with GELU activation.

    Projects from `in_dim` (ViT output) to `out_dim` (LLM hidden size).
    Optionally performs spatial pooling before projection (pool_factor > 1).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        pool_factor: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim + out_dim) // 2

        self.pool_factor = pool_factor

        # Spatial average pooling along the sequence dimension
        if pool_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor)
        else:
            self.pool = None

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (total_patches, in_dim) — raw ViT patch features

        Returns:
            (total_tokens, out_dim) — projected features ready for LLM
        """
        if self.pool is not None:
            # pool along sequence dim: (N, in_dim) → transpose → (1, in_dim, N)
            x = x.transpose(0, 1).unsqueeze(0)  # (1, in_dim, N)
            x = self.pool(x)                    # (1, in_dim, N // pool_factor)
            x = x.squeeze(0).transpose(0, 1)   # (N // pool_factor, in_dim)

        return self.net(x)
