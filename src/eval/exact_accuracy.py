from __future__ import annotations

import torch
from transformers import Trainer

IGNORE_LABEL_ID = -100


def exact_accuracy_from_logits(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_LABEL_ID
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    matches = (preds == labels) | ~mask
    per_sample = matches.all(dim=-1).float()
    return per_sample.mean()


def token_accuracy_from_logits(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_LABEL_ID
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    correct = (preds == labels) & mask
    return correct.float().sum() / mask.sum()


class ExactAccuracyEvaluator:
    def __init__(self, trainer: Trainer, pad_token_id: int):
        self._trainer = trainer
        self._pad_token_id = pad_token_id

    def teacher_forced(self, dataloader) -> float:
        model = self._trainer.model
        model.eval()
        exact_sum = 0.0
        token_sum = 0.0
        batches = 0
        for batch in dataloader:
            batch = self._trainer._prepare_inputs(batch)
            with torch.no_grad():
                outputs = model(**batch)
            exact_sum += float(exact_accuracy_from_logits(batch["labels"], outputs.logits))
            token_sum += float(token_accuracy_from_logits(batch["labels"], outputs.logits))
            batches += 1
        return exact_sum / max(1, batches), token_sum / max(1, batches)

    def autoregressive(self, dataloader) -> float:
        model = self._trainer.model
        model.eval()
        total = 0
        exact_matches = 0
        token_correct = 0
        token_total = 0

        for batch in dataloader:
            batch = self._trainer._prepare_inputs(batch)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            puzzle_identifiers = batch.get("puzzle_identifiers")

            with torch.no_grad():
                prefix_mask = labels == IGNORE_LABEL_ID
                prefix_lens = prefix_mask.sum(dim=1)
                max_prefix_len = int(prefix_lens.max().item())
                prefix_input_ids = input_ids[:, :max_prefix_len].clone()
                prefix_attention_mask = prefix_mask[:, :max_prefix_len].long()
                prefix_input_ids[prefix_attention_mask == 0] = self._pad_token_id

                target_tokens = [row[row != IGNORE_LABEL_ID] for row in labels]
                max_tgt_len = max(t.numel() for t in target_tokens)

                generated = model.generate(
                    input_ids=prefix_input_ids,
                    attention_mask=prefix_attention_mask,
                    max_new_tokens=max_tgt_len,
                    do_sample=False,
                    use_cache=True,
                    puzzle_identifiers=puzzle_identifiers,
                )

                for i, tgt in enumerate(target_tokens):
                    total += 1
                    if tgt.numel() == 0:
                        continue
                    start = int(prefix_lens[i].item())
                    end = start + int(tgt.numel())
                    pred = generated[i, start:end]
                    if torch.equal(pred, tgt):
                        exact_matches += 1
                    token_correct += int((pred == tgt).sum().item())
                    token_total += int(tgt.numel())

        exact = exact_matches / max(1, total)
        token_acc = token_correct / max(1, token_total)
        return exact, token_acc
