from __future__ import annotations

import torch
import torch.nn.functional as F

from .stablemax import stablemax_cross_entropy
from contextlib import nullcontext

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
    if getattr(logits, "use_stablemax", False):
        loss = stablemax_cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
        ).mean()
    else:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
        )
    return loss + aux_loss


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


def run_steps(
    steps: int,
    state,
    step_fn,
    *,
    training: bool,
    tbptt_steps: int,
    step_counter: int = 0,
    no_grad: bool = False,
    detach_fn=detach_state,
):
    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        for _ in range(int(steps)):
            state = step_fn(state)
            step_counter += 1
            if training and tbptt_steps > 0 and step_counter % tbptt_steps == 0:
                state = detach_fn(state)
    return state, step_counter


def run_act_steps(
    act_steps: int,
    state,
    act_step_fn,
    *,
    training: bool,
    tbptt_steps: int,
):
    act_steps = int(act_steps)
    if act_steps <= 0:
        return state
    if not training or tbptt_steps <= 0:
        start_grad = 0
    else:
        start_grad = max(act_steps - int(tbptt_steps), 0)
    for act_step in range(act_steps):
        force_no_grad = training and tbptt_steps > 0 and act_step < start_grad
        state = act_step_fn(state, force_no_grad)
    return state


def prepare_kv_cache(past_key_values, use_cache: bool, causal: bool):
    cache_enabled = bool(use_cache and causal)
    legacy_past = past_key_values
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        legacy_past = [(k, v) for k, v, _ in past_key_values]
    new_past = [] if cache_enabled else None
    return cache_enabled, legacy_past, new_past
