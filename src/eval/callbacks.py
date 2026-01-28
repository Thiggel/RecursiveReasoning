from __future__ import annotations

from transformers import TrainerCallback

from .exact_accuracy import ExactAccuracyEvaluator


class ExactAccuracyCallback(TrainerCallback):
    def __init__(self, task_name: str, pad_token_id: int):
        self._task_name = task_name
        self._pad_token_id = pad_token_id

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]
        evaluator = ExactAccuracyEvaluator(trainer, self._pad_token_id)

        train_loader = trainer.get_train_dataloader()
        eval_loader = trainer.get_eval_dataloader()

        if self._task_name == "causal":
            train_acc = evaluator.autoregressive(train_loader)
            eval_acc = evaluator.autoregressive(eval_loader)
        else:
            train_acc = evaluator.teacher_forced(train_loader)
            eval_acc = evaluator.teacher_forced(eval_loader)

        trainer.log({
            "train_exact_accuracy": train_acc,
            "eval_exact_accuracy": eval_acc,
        })
