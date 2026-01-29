from __future__ import annotations

import torch
import torch.nn as nn

from .config import BaseTransformerConfig
from .postnorm_block import PostNormBlock


class BlockStack(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig, num_layers: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList([PostNormBlock(cfg) for _ in range(self.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        present = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            x, pkv = layer(x, attention_mask, past_key_value=past, use_cache=use_cache)
            if use_cache:
                present.append(pkv)
        return x, present
