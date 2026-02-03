from __future__ import annotations

import torch
import torch.nn as nn

from .config import BaseTransformerConfig


class TokenAndPuzzleEmbedding(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.puzzle = nn.Embedding(cfg.num_puzzle_ids, cfg.d_model) if cfg.use_puzzle_emb else None
        self._causal = bool(cfg.causal)

    def forward(self, input_ids: torch.Tensor, puzzle_identifiers: torch.Tensor | None) -> torch.Tensor:
        h = self.tok(input_ids)
        if self.puzzle is not None:
            if puzzle_identifiers is None:
                raise ValueError("use_puzzle_emb=True but puzzle_identifiers is None")
            pos = 1 if self._causal else 0
            if h.shape[1] <= pos:
                raise ValueError("sequence too short for puzzle embedding position")
            h[:, pos:pos + 1, :] = self.puzzle(puzzle_identifiers)[:, None, :]
        return h
