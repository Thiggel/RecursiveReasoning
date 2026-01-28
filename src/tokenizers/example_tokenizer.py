from __future__ import annotations

from typing import Any

from .grid import GridPadder
from .vocabulary import TokenVocabulary


class ExampleTokenizer:
    IGNORE_LABEL_ID = -100
    def __init__(self, vocab: TokenVocabulary, grid_padder: GridPadder | None):
        self._vocab = vocab
        self._grid_padder = grid_padder

    @property
    def vocab_size(self) -> int:
        return self._vocab.vocab_size

    @property
    def special_ids(self) -> dict[str, int]:
        sp = self._vocab.special_ids
        return {"pad": sp.pad, "bos": sp.bos, "sep": sp.sep, "eos": sp.eos}

    @classmethod
    def from_config(cls, cfg_vocab) -> "ExampleTokenizer":
        vocab = TokenVocabulary.from_config(cfg_vocab)
        grid_padder = GridPadder.from_config(cfg_vocab)
        return cls(vocab=vocab, grid_padder=grid_padder)

    def encode_inputs_labels(self, example: dict[str, Any]) -> tuple[list[int], list[int], int]:
        inputs = example["inputs"]
        labels = example["labels"]

        inputs = self._prepare_grid(inputs)
        labels = self._prepare_grid(labels)

        input_ids = self._flatten_and_encode(inputs)
        label_ids = self._flatten_and_encode(labels)
        puzzle_identifier = int(example.get("puzzle_identifier", 0))

        return input_ids, label_ids, puzzle_identifier

    def make_mapper(self, task_name: str, include_puzzle: bool):
        special = self.special_ids
        puzzle_token_id = self._puzzle_token_id(special) if include_puzzle else None

        def map_encoder(ex: dict[str, Any]) -> dict[str, Any]:
            input_ids, label_ids, puzzle_identifier = self.encode_inputs_labels(ex)
            if include_puzzle:
                input_ids = [puzzle_token_id] + input_ids
                labels = [self.IGNORE_LABEL_ID] + label_ids
            else:
                labels = label_ids
            labels = [t if t != special["pad"] else self.IGNORE_LABEL_ID for t in labels]
            item = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
            if include_puzzle:
                item["puzzle_identifiers"] = puzzle_identifier
            return item

        def map_causal(ex: dict[str, Any]) -> dict[str, Any]:
            input_ids, label_ids, puzzle_identifier = self.encode_inputs_labels(ex)
            if include_puzzle:
                input_prefix = [special["bos"], puzzle_token_id] + input_ids + [special["sep"]]
            else:
                input_prefix = [special["bos"]] + input_ids + [special["sep"]]
            input_ids = input_prefix + label_ids + [special["eos"]]
            prefix_len = len(input_prefix)
            labels = [self.IGNORE_LABEL_ID] * prefix_len + label_ids + [special["eos"]]
            item = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
            if include_puzzle:
                item["puzzle_identifiers"] = puzzle_identifier
            return item

        return map_causal if task_name == "causal" else map_encoder

    def _prepare_grid(self, grid: Any):
        if self._grid_padder is None:
            return grid
        if self._is_nested_list(grid):
            return self._grid_padder.pad(grid)
        return grid

    def _flatten_and_encode(self, values: Any) -> list[int]:
        if self._is_nested_list(values):
            return [self._vocab.encode_value(v) for row in values for v in row]
        return [self._vocab.encode_value(v) for v in values]

    @staticmethod
    def _is_nested_list(values: Any) -> bool:
        return isinstance(values, list) and len(values) > 0 and isinstance(values[0], list)

    def _puzzle_token_id(self, special: dict[str, int]) -> int:
        return special["pad"]
