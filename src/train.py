from __future__ import annotations

import hydra
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from src.data import PadCollator
from src.eval import ExactAccuracyCallback
from src.models import get_model_class
from src.tokenizers import ExampleTokenizer


def _merge_config(model_cfg, vocab_cfg) -> dict[str, object]:
    model_dict = OmegaConf.to_container(model_cfg, resolve=True)
    vocab_dict = OmegaConf.to_container(vocab_cfg, resolve=True)
    merged: dict[str, object] = {}
    if isinstance(vocab_dict, dict):
        merged.update(vocab_dict)
    if isinstance(model_dict, dict):
        merged.update(model_dict)
    return merged


def _build_tokenizer(cfg) -> tuple[ExampleTokenizer, dict[str, int], int, int]:
    tokenizer = ExampleTokenizer.from_config(cfg.vocab)
    special_ids = tokenizer.special_ids
    vocab_size = tokenizer.vocab_size

    if not cfg.model.use_puzzle_emb:
        return tokenizer, special_ids, vocab_size, int(cfg.model.num_puzzle_ids)

    if cfg.model.num_puzzle_ids != -1:
        return tokenizer, special_ids, vocab_size, int(cfg.model.num_puzzle_ids)

    return tokenizer, special_ids, vocab_size, -1


def _prepare_dataset(cfg, tokenizer: ExampleTokenizer) -> tuple[dict, int]:
    ds = load_dataset(cfg.vocab.path)

    include_puzzle = bool(cfg.model.use_puzzle_emb)
    mapper = tokenizer.make_mapper(cfg.task.name, include_puzzle=include_puzzle)
    remove_cols = ds["train"].column_names
    ds_tok = ds.map(mapper, remove_columns=remove_cols)

    num_puzzle_ids = int(cfg.model.num_puzzle_ids)
    if include_puzzle and cfg.model.num_puzzle_ids == -1:
        max_id = max(ds["train"]["puzzle_identifier"]) if "puzzle_identifier" in ds["train"].column_names else 0
        num_puzzle_ids = int(max_id) + 1

    return ds_tok, num_puzzle_ids


def _build_model(cfg, vocab_size: int, special_ids: dict[str, int], num_puzzle_ids: int):
    merged_cfg = _merge_config(cfg.model, cfg.vocab)
    merged_cfg.update({
        "vocab_size": vocab_size,
        "pad_token_id": special_ids["pad"],
        "bos_token_id": special_ids["bos"],
        "sep_token_id": special_ids["sep"],
        "eos_token_id": special_ids["eos"],
        "num_puzzle_ids": num_puzzle_ids,
    })
    model_cls = get_model_class(cfg.model.name)
    model_cfg = model_cls.config_class(**merged_cfg)
    return model_cls(model_cfg)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    tokenizer, special_ids, vocab_size, num_puzzle_ids = _build_tokenizer(cfg)
    ds_tok, inferred_puzzle_ids = _prepare_dataset(cfg, tokenizer)
    if cfg.model.use_puzzle_emb and cfg.model.num_puzzle_ids == -1:
        num_puzzle_ids = inferred_puzzle_ids
        print(f"[info] computed num_puzzle_ids={num_puzzle_ids}")

    model = _build_model(cfg, vocab_size, special_ids, num_puzzle_ids)

    eval_dataset = ds_tok["test"] if "test" in ds_tok else None
    max_eval_samples = int(cfg.train.max_eval_samples) if "max_eval_samples" in cfg.train else None
    if eval_dataset is not None and max_eval_samples is not None and max_eval_samples > 0:
        max_eval_samples = min(max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    train_args = OmegaConf.to_container(cfg.train, resolve=True)
    train_args.pop("max_eval_samples", None)
    if eval_dataset is None:
        train_args["eval_strategy"] = None
    args = TrainingArguments(**train_args)

    collator = PadCollator(pad_id=special_ids["pad"])

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.add_callback(ExactAccuracyCallback(cfg.task.name, special_ids["pad"], trainer=trainer))

    trainer.train()
    trainer.save_model(cfg.train.output_dir)


if __name__ == "__main__":
    main()
