from __future__ import annotations

import torch


def _s(x: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1 / (1 - x + eps), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = _s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    valid_mask = labels != ignore_index
    labels_safe = torch.where(valid_mask, labels, 0)
    pred_logprobs = torch.gather(logprobs, index=labels_safe.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid_mask, pred_logprobs, torch.zeros_like(pred_logprobs))
