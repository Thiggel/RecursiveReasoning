from __future__ import annotations

from typing import Iterable


class TokenVocabulary:
    DEFAULT_SPECIAL_IDS = {"pad": 0, "bos": 1, "sep": 2, "eos": 3}
    DEFAULT_NUMERIC_TOKENS = tuple(range(10))

    def __init__(self, token_to_id: dict[object, int], special_ids: dict[str, int]):
        self._token_to_id = token_to_id
        self._token_to_id_str = {str(k): v for k, v in token_to_id.items()}
        self._special_ids = special_ids
        self._vocab_size = self._compute_vocab_size()

    @property
    def special_ids(self) -> dict[str, int]:
        return self._special_ids

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode_value(self, value: object) -> int:
        if value in self._token_to_id:
            return self._token_to_id[value]
        value_str = str(value)
        if value_str in self._token_to_id_str:
            return self._token_to_id_str[value_str]
        raise KeyError(f"Unknown token: {value}")

    @classmethod
    def from_config(cls, cfg_vocab) -> "TokenVocabulary":
        special_ids = cls._special_ids_from_config(cfg_vocab)
        token_to_id = cls._tokens_from_config(cfg_vocab, special_ids)
        return cls(token_to_id=token_to_id, special_ids=special_ids)

    @classmethod
    def _special_ids_from_config(cls, cfg_vocab) -> dict[str, int]:
        if "special_ids" not in cfg_vocab:
            return cls.DEFAULT_SPECIAL_IDS
        sp = cfg_vocab.special_ids
        if isinstance(sp, dict):
            return {
                "pad": int(sp["pad"]),
                "bos": int(sp["bos"]),
                "sep": int(sp["sep"]),
                "eos": int(sp["eos"]),
            }
        return {"pad": int(sp.pad), "bos": int(sp.bos), "sep": int(sp.sep), "eos": int(sp.eos)}

    @classmethod
    def _tokens_from_config(cls, cfg_vocab, special_ids: dict[str, int]) -> dict[object, int]:
        if "tokens" in cfg_vocab:
            raw = cfg_vocab.tokens
            if isinstance(raw, dict):
                return {k: int(v) for k, v in raw.items()}
            return cls._assign_sequential(raw, cls._first_non_special_id(special_ids))
        return cls._assign_sequential(cls.DEFAULT_NUMERIC_TOKENS, cls._first_non_special_id(special_ids))

    @staticmethod
    def _assign_sequential(tokens: Iterable[object], start_id: int) -> dict[object, int]:
        return {token: start_id + i for i, token in enumerate(tokens)}

    @staticmethod
    def _first_non_special_id(special_ids: dict[str, int]) -> int:
        return max(special_ids["pad"], special_ids["bos"], special_ids["sep"], special_ids["eos"]) + 1

    def _compute_vocab_size(self) -> int:
        max_token_id = max(self._token_to_id.values()) if self._token_to_id else -1
        max_special_id = max(self._special_ids["pad"], self._special_ids["bos"], self._special_ids["sep"], self._special_ids["eos"])
        return max(max_token_id, max_special_id) + 1
