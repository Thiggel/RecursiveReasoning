from __future__ import annotations

from typing import Sequence


class GridPadder:
    DEFAULT_RAW_PAD_VALUE = 0

    def __init__(self, height: int, width: int, pad_value: int):
        self._height = height
        self._width = width
        self._pad_value = pad_value

    @classmethod
    def from_config(cls, cfg_vocab):
        if "grid" not in cfg_vocab:
            return None
        height = int(cfg_vocab.grid.H)
        width = int(cfg_vocab.grid.W)
        pad_value = int(getattr(cfg_vocab, "raw_pad_value", cls.DEFAULT_RAW_PAD_VALUE))
        return cls(height=height, width=width, pad_value=pad_value)

    def pad(self, grid: Sequence[Sequence[int]]) -> list[list[int]]:
        height = len(grid)
        width = len(grid[0]) if height else 0
        padded = [list(row) + [self._pad_value] * (self._width - width) for row in grid]
        padded += [[self._pad_value] * self._width for _ in range(self._height - height)]
        return padded
