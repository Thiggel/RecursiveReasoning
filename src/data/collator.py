from __future__ import annotations

from dataclasses import dataclass
import torch

IGNORE_LABEL_ID = -100


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long),
            "labels": torch.tensor([pad(f["labels"], IGNORE_LABEL_ID) for f in features], dtype=torch.long),
        }
        if "puzzle_identifiers" in features[0]:
            batch["puzzle_identifiers"] = torch.tensor([f["puzzle_identifiers"] for f in features], dtype=torch.long)
        return batch
