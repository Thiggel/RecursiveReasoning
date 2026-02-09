#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import Dataset, DatasetDict


def _iter_split(inputs: np.ndarray, labels: np.ndarray, *, reshape: bool):
    for i in range(inputs.shape[0]):
        inp = inputs[i]
        lab = labels[i]
        if reshape:
            inp = inp.reshape(9, 9)
            lab = lab.reshape(9, 9)
        yield {
            "inputs": inp.tolist(),
            "labels": lab.tolist(),
            "puzzle_index": int(i),
            "group_index": int(i),
            "set": "all",
        }


def _load_split(tiny_dir: str, split: str):
    split_dir = os.path.join(tiny_dir, split)
    inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")
    return inputs, labels


def main():
    parser = argparse.ArgumentParser(description="Convert TinyRecursiveModels Sudoku .npy dataset to HF dataset.")
    parser.add_argument("--tiny-dir", required=True, help="TinyRecursiveModels dataset directory (contains train/test).")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF dataset (save_to_disk).")
    parser.add_argument("--keep-flat", action="store_true", help="Keep inputs/labels as flat length-81 lists.")
    args = parser.parse_args()

    train_inputs, train_labels = _load_split(args.tiny_dir, "train")
    test_inputs, test_labels = _load_split(args.tiny_dir, "test")

    reshape = not args.keep_flat
    train_ds = Dataset.from_generator(lambda: _iter_split(train_inputs, train_labels, reshape=reshape))
    test_ds = Dataset.from_generator(lambda: _iter_split(test_inputs, test_labels, reshape=reshape))

    os.makedirs(args.output_dir, exist_ok=True)
    DatasetDict({"train": train_ds, "test": test_ds}).save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
