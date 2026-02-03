from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention
from .config import BaseTransformerConfig
from .convswiglu import ConvSwiGLU, SwiGLU


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_sq + self.eps) * self.weight


class PostNormBlock(nn.Module):
    """
    Post-norm:
      x = LN(x + Attn(x))
      x = LN(x + MLP(x))
    """
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm1 = RMSNorm(cfg.d_model)

        if cfg.use_convswiglu:
            self.mlp = ConvSwiGLU(
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                kernel_size=cfg.convswiglu_kernel_size,
                groups=cfg.convswiglu_groups,
            )
        else:
            self.mlp = SwiGLU(
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
            )
        self.norm2 = RMSNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(x, attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x, present
