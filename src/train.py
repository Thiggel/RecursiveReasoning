from __future__ import annotations

import hydra
import os

from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from src.data import PadCollator
from src.eval.exact_accuracy import exact_accuracy_from_logits, token_accuracy_from_logits
from src.models import get_model_class
from src.tokenizers import ExampleTokenizer


class AccuracyTrainer(Trainer):
    """
    Computes per-batch exact and token accuracy during both training and evaluation,
    logging them with `train_*/eval_*` prefixes. This avoids the extra full-dataset
    passes that the previous callback performed.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels")
        if logits is not None and labels is not None:
            exact = exact_accuracy_from_logits(labels, logits)
            token = token_accuracy_from_logits(labels, logits)
            prefix = "train" if model.training else "eval"
            # log() respects the Trainer's logging callbacks; per-batch metrics are emitted here.
            self.log({f"{prefix}_exact_accuracy": float(exact), f"{prefix}_token_accuracy": float(token)})

        return (loss, outputs) if return_outputs else loss


def _merge_config(model_cfg, vocab_cfg) -> dict[str, object]:
    model_dict = OmegaConf.to_container(model_cfg, resolve=True)
    vocab_dict = OmegaConf.to_container(vocab_cfg, resolve=True)
    merged: dict[str, object] = {}
    if isinstance(vocab_dict, dict):
        merged.update(vocab_dict)
    if isinstance(model_dict, dict):
        merged.update(model_dict)
    if isinstance(vocab_dict, dict) and "use_puzzle_emb" in vocab_dict:
        merged["use_puzzle_emb"] = vocab_dict["use_puzzle_emb"]
    return merged


def _build_tokenizer(cfg) -> tuple[ExampleTokenizer, dict[str, int], int, int]:
    tokenizer = ExampleTokenizer.from_config(cfg.vocab)
    special_ids = tokenizer.special_ids
    vocab_size = tokenizer.vocab_size
    include_puzzle = bool(getattr(cfg.vocab, "use_puzzle_emb", cfg.model.use_puzzle_emb))
    if not include_puzzle:
        return tokenizer, special_ids, vocab_size, int(cfg.model.num_puzzle_ids)

    if cfg.model.num_puzzle_ids != -1:
        return tokenizer, special_ids, vocab_size, int(cfg.model.num_puzzle_ids)

    return tokenizer, special_ids, vocab_size, -1


def _load_dataset(path: str):
    if os.path.isdir(path):
        dataset_dict = os.path.join(path, "dataset_dict.json")
        if os.path.exists(dataset_dict):
            return load_from_disk(path)
    return load_dataset(path)


def _prepare_dataset(cfg, tokenizer: ExampleTokenizer) -> tuple[dict, int]:
    ds = _load_dataset(cfg.vocab.path)

    include_puzzle = bool(getattr(cfg.vocab, "use_puzzle_emb", cfg.model.use_puzzle_emb))
    mapper = tokenizer.make_mapper(cfg.task.name, include_puzzle=include_puzzle)
    remove_cols = ds["train"].column_names
    ds_tok = ds.map(mapper, remove_columns=remove_cols)

    num_puzzle_ids = int(cfg.model.num_puzzle_ids)
    if include_puzzle and cfg.model.num_puzzle_ids == -1:
        max_id = max(ds["train"]["puzzle_identifier"]) if "puzzle_identifier" in ds["train"].column_names else 0
        num_puzzle_ids = int(max_id) + 1

    return ds_tok, num_puzzle_ids


def _debug_samples(cfg, tokenizer: ExampleTokenizer, ds_raw):
    debug_samples = int(getattr(cfg.train, "debug_samples", 0))
    if debug_samples <= 0:
        return
    vocab = tokenizer._vocab
    include_puzzle = bool(getattr(cfg.vocab, "use_puzzle_emb", cfg.model.use_puzzle_emb))
    print(f"[debug] printing {debug_samples} raw/tokenized samples (include_puzzle={include_puzzle})")
    for idx in range(min(debug_samples, len(ds_raw["train"]))):
        ex = ds_raw["train"][idx]
        input_ids, label_ids, puzzle_identifier = tokenizer.encode_inputs_labels(ex)
        if include_puzzle:
            puzzle_token_id = tokenizer._puzzle_token_id(tokenizer.special_ids)
            input_ids = [puzzle_token_id] + input_ids
            label_ids = [ExampleTokenizer.IGNORE_LABEL_ID] + label_ids
        decoded_inputs = vocab.decode_ids(input_ids)
        decoded_labels = [
            "<ignore>" if t == ExampleTokenizer.IGNORE_LABEL_ID else vocab.decode_id(int(t))
            for t in label_ids
        ]
        print(f"[debug][{idx}] puzzle_id={puzzle_identifier}")
        print(f"[debug][{idx}] inputs_raw={ex['inputs']}")
        print(f"[debug][{idx}] labels_raw={ex['labels']}")
        print(f"[debug][{idx}] input_ids={input_ids}")
        print(f"[debug][{idx}] label_ids={label_ids}")
        print(f"[debug][{idx}] decoded_inputs={decoded_inputs}")
        print(f"[debug][{idx}] decoded_labels={decoded_labels}")


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
    ds = _load_dataset(cfg.vocab.path)
    _debug_samples(cfg, tokenizer, ds)
    ds_tok, inferred_puzzle_ids = _prepare_dataset(cfg, tokenizer)
    if cfg.model.use_puzzle_emb and cfg.model.num_puzzle_ids == -1:
        num_puzzle_ids = inferred_puzzle_ids
        print(f"[info] computed num_puzzle_ids={num_puzzle_ids}")

    model = _build_model(cfg, vocab_size, special_ids, num_puzzle_ids)
    if bool(getattr(cfg.train, "compile_model", False)):
        import torch
        model = torch.compile(model, dynamic=False)

    eval_dataset = ds_tok["test"] if "test" in ds_tok else None
    max_eval_samples = int(cfg.train.max_eval_samples) if "max_eval_samples" in cfg.train else None
    if eval_dataset is not None and max_eval_samples is not None and max_eval_samples > 0:
        max_eval_samples = min(max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    train_args = OmegaConf.to_container(cfg.train, resolve=True)
    train_args.pop("max_eval_samples", None)
    train_args.pop("debug_samples", None)
    train_args.pop("compile_model", None)
    # torch.compile wraps the model in OptimizedModule whose forward signature is (*args, **kwargs),
    # which breaks Trainer's unused column pruning. Keep all columns when compiling unless explicitly set.
    if bool(getattr(cfg.train, "compile_model", False)):
        train_args.setdefault("remove_unused_columns", False)
    if eval_dataset is None:
        train_args["eval_strategy"] = None
    args = TrainingArguments(**train_args)

    collator = PadCollator(pad_id=special_ids["pad"])

    trainer = AccuracyTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.train.output_dir)


if __name__ == "__main__":
    main()
