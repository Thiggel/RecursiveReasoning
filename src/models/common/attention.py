from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BaseTransformerConfig
from .rope import RoPE, _apply_rope


class Attention(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
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
        q, k, v = self._project_qkv(x)
        past_len = self._past_length(past_key_value)
        q, k = self._apply_rope(q, k, x.device, past_len)
        k, v = self._apply_past(k, v, past_key_value)
        attn_mask, use_causal_only = self._build_attn_mask(attention_mask, q.shape[-2], k.shape[-2], past_len, x.device)
        y = self._attend(q, k, v, attn_mask, use_causal_only)
        y = self._merge_heads(y, x.shape[2])
        y = self.proj(y)
        y = self.drop(y)
        present = (k, v) if use_cache else None
        return y, present

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.H, self.Dh).transpose(1, 2)
        k = k.view(B, L, self.H, self.Dh).transpose(1, 2)
        v = v.view(B, L, self.H, self.Dh).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _past_length(past_key_value: tuple[torch.Tensor, torch.Tensor] | None) -> int:
        if past_key_value is None:
            return 0
        return past_key_value[0].shape[-2]

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, device: torch.device, past_len: int):
        cos, sin = self.rope(q.shape[-2], device, offset=past_len)
        return _apply_rope(q, k, cos, sin)

    @staticmethod
    def _apply_past(
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_value is None:
            return k, v
        k = torch.cat([past_key_value[0], k], dim=-2)
        v = torch.cat([past_key_value[1], v], dim=-2)
        return k, v

    def _build_attn_mask(
        self,
        attention_mask: torch.Tensor | None,
        query_len: int,
        total_len: int,
        past_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, bool]:
        key_padding_mask = None
        has_padding = False
        if attention_mask is not None:
            if past_len > 0 and attention_mask.shape[1] == query_len:
                past_pad = torch.ones((attention_mask.shape[0], past_len), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([past_pad, attention_mask], dim=1)
            key_padding_mask = (attention_mask == 0)
            has_padding = bool(key_padding_mask.any().item())

        use_causal_only = self.cfg.causal and not has_padding
        attn_mask = None
        if not use_causal_only:
            if self.cfg.causal:
                attn_mask = torch.triu(
                    torch.ones(query_len, total_len, device=device, dtype=torch.bool),
                    diagonal=1 + past_len,
                )
            if key_padding_mask is not None:
                pad_mask = key_padding_mask[:, None, None, :]
                if attn_mask is None:
                    attn_mask = pad_mask
                else:
                    attn_mask = attn_mask[None, None, :, :] | pad_mask
        return attn_mask, use_causal_only

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        use_causal_only: bool,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=use_causal_only,
        )

    @staticmethod
    def _merge_heads(y: torch.Tensor, d_model: int) -> torch.Tensor:
        y = y.transpose(1, 2).contiguous()
        return y.view(y.shape[0], y.shape[1], d_model)
