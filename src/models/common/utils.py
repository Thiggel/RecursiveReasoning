from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_LABEL_ID = -100


def init_state_like(x: torch.Tensor, mode: str, sigma: float) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "noise":
        return torch.randn_like(x) * float(sigma)
    raise ValueError(f"Unknown state_init: {mode}")


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    aux_loss: torch.Tensor,
) -> torch.Tensor | None:
    if labels is None:
        return None
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_LABEL_ID,
    ) + aux_loss


def detach_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_state(s) for s in state)
    if isinstance(state, list):
        return [detach_state(s) for s in state]
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    return state


def prepare_kv_cache(past_key_values, use_cache: bool, causal: bool):
    cache_enabled = bool(use_cache and causal)
    legacy_past = past_key_values
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        legacy_past = [(k, v) for k, v, _ in past_key_values]
    new_past = [] if cache_enabled else None
    return cache_enabled, legacy_past, new_past
