from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention
from .config import BaseTransformerConfig
from .convswiglu import ConvSwiGLU


class PostNormBlock(nn.Module):
    """
    Post-norm:
      x = LN(x + Attn(x))
      x = LN(x + MLP(x))
    """
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)

        if cfg.use_convswiglu:
            self.mlp = ConvSwiGLU(
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                kernel_size=cfg.convswiglu_kernel_size,
                groups=cfg.convswiglu_groups,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(x, attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x, present
