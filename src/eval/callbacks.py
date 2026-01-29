from __future__ import annotations

from transformers import TrainerCallback

from .exact_accuracy import ExactAccuracyEvaluator


class ExactAccuracyCallback(TrainerCallback):
    def __init__(self, task_name: str, pad_token_id: int, trainer=None):
        self._task_name = task_name
        self._pad_token_id = pad_token_id
        self._trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer") or self._trainer
        if trainer is None:
            raise ValueError("ExactAccuracyCallback requires a trainer instance")
        evaluator = ExactAccuracyEvaluator(trainer, self._pad_token_id)

        train_loader = trainer.get_train_dataloader()
        eval_loader = trainer.get_eval_dataloader()

        if self._task_name == "causal":
            train_exact, train_token = evaluator.autoregressive(train_loader)
            eval_exact, eval_token = evaluator.autoregressive(eval_loader)
        else:
            train_exact, train_token = evaluator.teacher_forced(train_loader)
            eval_exact, eval_token = evaluator.teacher_forced(eval_loader)

        trainer.log({
            "train_exact_accuracy": train_exact,
            "eval_exact_accuracy": eval_exact,
            "train_token_accuracy": train_token,
            "eval_token_accuracy": eval_token,
        })
