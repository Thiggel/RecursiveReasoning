#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def _load_csv_split(source_repo: str, split: str, min_difficulty: int | None):
    inputs = []
    labels = []
    csv_path = hf_hub_download(source_repo, f"{split}.csv", repo_type="dataset")
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, q, a, rating in reader:
            if min_difficulty is not None and int(rating) < min_difficulty:
                continue
            if len(q) != 81 or len(a) != 81:
                raise ValueError("Expected 81-char Sudoku strings.")
            inputs.append(np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
    return inputs, labels


def _iter_split(
    split: str,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    *,
    num_aug: int,
    plus_one: bool,
):
    puzzle_index = 0
    group_index = 0
    aug_count = num_aug if split == "train" else 0
    for orig_inp, orig_out in zip(inputs, labels):
        for aug_idx in range(1 + aug_count):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)
            if plus_one:
                inp = inp + 1
                out = out + 1
            if inp.min() < (1 if plus_one else 0) or inp.max() > (10 if plus_one else 9):
                raise RuntimeError(f"Input range out of bounds: [{inp.min()},{inp.max()}].")
            if out.min() < (1 if plus_one else 0) or out.max() > (10 if plus_one else 9):
                raise RuntimeError(f"Label range out of bounds: [{out.min()},{out.max()}].")
            yield {
                "inputs": inp.tolist(),
                "labels": out.tolist(),
                "puzzle_index": int(puzzle_index),
                "group_index": int(group_index),
                "set": "all",
            }
            puzzle_index += 1
        group_index += 1


def main():
    parser = argparse.ArgumentParser(description="Build TinyRecursiveModels-style Sudoku dataset and save as HF dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the HF dataset (save_to_disk).")
    parser.add_argument("--source-repo", default="sapientinc/sudoku-extreme", help="Hugging Face dataset repo.")
    parser.add_argument("--subsample-size", type=int, default=1000, help="Train subsample size (Tiny default: 1000).")
    parser.add_argument("--num-aug", type=int, default=1000, help="Train augmentations per example (Tiny default: 1000).")
    parser.add_argument("--min-difficulty", type=int, default=None, help="Minimum difficulty rating filter.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument("--no-plus-one", action="store_true", help="Disable +1 shift (keep raw 0..9 digits).")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    train_inputs, train_labels = _load_csv_split(args.source_repo, "train", args.min_difficulty)
    test_inputs, test_labels = _load_csv_split(args.source_repo, "test", args.min_difficulty)

    if args.subsample_size is not None and args.subsample_size < len(train_inputs):
        indices = np.random.choice(len(train_inputs), size=args.subsample_size, replace=False)
        train_inputs = [train_inputs[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    plus_one = not args.no_plus_one

    train_ds = Dataset.from_generator(
        lambda: _iter_split(
            "train",
            train_inputs,
            train_labels,
            num_aug=args.num_aug,
            plus_one=plus_one,
        )
    )
    test_ds = Dataset.from_generator(
        lambda: _iter_split(
            "test",
            test_inputs,
            test_labels,
            num_aug=0,
            plus_one=plus_one,
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    DatasetDict({"train": train_ds, "test": test_ds}).save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
