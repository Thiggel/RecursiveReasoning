from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(gate) * value
        x = self.out_proj(x)
        return self.drop(x)


class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        kernel_size: int = 3,
        groups: int | None = None,
    ):
        super().__init__()
        conv_groups = groups if groups is not None else d_model
        padding = kernel_size // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=conv_groups,
        )
        self.in_proj = nn.Linear(d_model, 2 * d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.depthwise_conv(h)
        h = h.transpose(1, 2)
        gate, value = self.in_proj(h).chunk(2, dim=-1)
        h = self.act(gate) * value
        h = self.out_proj(h)
        h = self.drop(h)
        return h
