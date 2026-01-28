from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig
from .rope import RoPE, _apply_rope


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.H = cfg.n_heads
        self.Dh = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RoPE(self.Dh, theta=cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.H, self.Dh).transpose(1, 2)
        k = k.view(B, L, self.H, self.Dh).transpose(1, 2)
        v = v.view(B, L, self.H, self.Dh).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[-2]

        cos, sin = self.rope(L, x.device, offset=past_len)
        q, k = _apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        total_len = past_len + L
        key_padding_mask = None
        has_padding = False
        if attention_mask is not None:
            if past_len > 0 and attention_mask.shape[1] == L:
                past_pad = torch.ones((attention_mask.shape[0], past_len), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([past_pad, attention_mask], dim=1)
            key_padding_mask = (attention_mask == 0)
            has_padding = bool(key_padding_mask.any().item())

        use_causal_only = self.cfg.causal and not has_padding
        attn_mask = None
        if not use_causal_only:
            if self.cfg.causal:
                attn_mask = torch.triu(
                    torch.ones(L, total_len, device=x.device, dtype=torch.bool),
                    diagonal=1 + past_len,
                )
            if key_padding_mask is not None:
                pad_mask = key_padding_mask[:, None, None, :]
                if attn_mask is None:
                    attn_mask = pad_mask
                else:
                    attn_mask = attn_mask[None, None, :, :] | pad_mask

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=use_causal_only,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.proj(y)
        y = self.drop(y)

        present = (k, v) if use_cache else None
        return y, present
