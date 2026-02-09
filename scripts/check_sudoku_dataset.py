#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


FORMAT_AUTO = "auto"
FORMAT_RAW = "raw"
FORMAT_PLUS_ONE = "plus_one"
ALLOWED_FORMATS = (FORMAT_AUTO, FORMAT_RAW, FORMAT_PLUS_ONE)


def _load_hf_dataset(path_or_name: str) -> DatasetDict:
    if os.path.isdir(path_or_name):
        dataset_dict_path = os.path.join(path_or_name, "dataset_dict.json")
        if os.path.exists(dataset_dict_path):
            return load_from_disk(path_or_name)
    loaded = load_dataset(path_or_name)
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict for {path_or_name}, got: {type(loaded)}")
    return loaded


def _reshape_to_grid(arr: np.ndarray, name: str, start_index: int) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[1:] == (9, 9):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 81:
        return arr.reshape(-1, 9, 9)
    raise ValueError(
        f"{name} at slice start {start_index} has unexpected shape {arr.shape}; expected (N,9,9) or (N,81)."
    )


def _infer_format(inputs: np.ndarray, labels: np.ndarray) -> str:
    in_min, in_max = int(inputs.min()), int(inputs.max())
    lb_min, lb_max = int(labels.min()), int(labels.max())
    if in_max > 9 or lb_max > 9:
        return FORMAT_PLUS_ONE
    if in_min == 0:
        return FORMAT_RAW
    if in_min >= 1 and lb_min >= 2:
        return FORMAT_PLUS_ONE
    return FORMAT_RAW


def _normalize(inputs: np.ndarray, labels: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    if fmt == FORMAT_RAW:
        in_norm = inputs.astype(np.int16, copy=False)
        lb_norm = labels.astype(np.int16, copy=False)
    elif fmt == FORMAT_PLUS_ONE:
        in_norm = inputs.astype(np.int16, copy=False) - 1
        lb_norm = labels.astype(np.int16, copy=False) - 1
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return in_norm, lb_norm


def _iter_hf_chunks(split_ds: Dataset, chunk_size: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    total = len(split_ds)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        batch = split_ds[start:end]
        inputs = _reshape_to_grid(np.asarray(batch["inputs"]), "inputs", start)
        labels = _reshape_to_grid(np.asarray(batch["labels"]), "labels", start)
        yield start, inputs, labels


def _iter_tiny_chunks(tiny_dir: str, split: str, chunk_size: int) -> tuple[int, Iterable[tuple[int, np.ndarray, np.ndarray]]]:
    split_dir = os.path.join(tiny_dir, split)
    in_path = os.path.join(split_dir, "all__inputs.npy")
    lb_path = os.path.join(split_dir, "all__labels.npy")
    if not os.path.exists(in_path) or not os.path.exists(lb_path):
        raise FileNotFoundError(f"Missing Tiny split files under {split_dir}")
    inputs = np.load(in_path, mmap_mode="r")
    labels = np.load(lb_path, mmap_mode="r")
    if inputs.shape[0] != labels.shape[0]:
        raise ValueError(f"Tiny split {split}: inputs/labels mismatch {inputs.shape[0]} vs {labels.shape[0]}")
    total = int(inputs.shape[0])

    def _iter():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            in_chunk = _reshape_to_grid(np.asarray(inputs[start:end]), "inputs", start)
            lb_chunk = _reshape_to_grid(np.asarray(labels[start:end]), "labels", start)
            yield start, in_chunk, lb_chunk

    return total, _iter()


def _append_first_indices(store: list[int], mask: np.ndarray, start: int, max_items: int) -> None:
    if len(store) >= max_items:
        return
    local = np.flatnonzero(mask)
    for idx in local[: max_items - len(store)]:
        store.append(int(start + idx))


@dataclass
class SplitAudit:
    split: str
    num_examples: int = 0
    format_used: str | None = None
    raw_input_min: int = 10**9
    raw_input_max: int = -(10**9)
    raw_label_min: int = 10**9
    raw_label_max: int = -(10**9)
    norm_input_min: int = 10**9
    norm_input_max: int = -(10**9)
    norm_label_min: int = 10**9
    norm_label_max: int = -(10**9)
    raw_input_values: set[int] = field(default_factory=set)
    raw_label_values: set[int] = field(default_factory=set)
    num_input_out_of_range_examples: int = 0
    num_label_out_of_range_examples: int = 0
    num_clue_mismatch_examples: int = 0
    num_invalid_row_examples: int = 0
    num_invalid_col_examples: int = 0
    num_invalid_box_examples: int = 0
    num_invalid_solution_examples: int = 0
    blank_count_total: int = 0
    blank_count_min: int = 10**9
    blank_count_max: int = -(10**9)
    digest_sha256: hashlib._hashlib.HASH = field(default_factory=hashlib.sha256, repr=False)
    first_bad_indices: dict[str, list[int]] = field(
        default_factory=lambda: {
            "input_out_of_range": [],
            "label_out_of_range": [],
            "clue_mismatch": [],
            "invalid_row": [],
            "invalid_col": [],
            "invalid_box": [],
            "invalid_solution": [],
        }
    )

    def to_json(self) -> dict:
        blank_mean = 0.0 if self.num_examples == 0 else float(self.blank_count_total) / float(self.num_examples)
        return {
            "split": self.split,
            "num_examples": self.num_examples,
            "format_used": self.format_used,
            "raw_input_range": [self.raw_input_min, self.raw_input_max],
            "raw_label_range": [self.raw_label_min, self.raw_label_max],
            "norm_input_range": [self.norm_input_min, self.norm_input_max],
            "norm_label_range": [self.norm_label_min, self.norm_label_max],
            "raw_input_values": sorted(self.raw_input_values),
            "raw_label_values": sorted(self.raw_label_values),
            "num_input_out_of_range_examples": self.num_input_out_of_range_examples,
            "num_label_out_of_range_examples": self.num_label_out_of_range_examples,
            "num_clue_mismatch_examples": self.num_clue_mismatch_examples,
            "num_invalid_row_examples": self.num_invalid_row_examples,
            "num_invalid_col_examples": self.num_invalid_col_examples,
            "num_invalid_box_examples": self.num_invalid_box_examples,
            "num_invalid_solution_examples": self.num_invalid_solution_examples,
            "blank_count_min": self.blank_count_min if self.blank_count_min != 10**9 else None,
            "blank_count_max": self.blank_count_max if self.blank_count_max != -(10**9) else None,
            "blank_count_mean": blank_mean,
            "sha256_inputs_labels_normalized_ordered": self.digest_sha256.hexdigest(),
            "first_bad_indices": self.first_bad_indices,
        }


def _audit_chunks(
    split: str,
    chunks: Iterable[tuple[int, np.ndarray, np.ndarray]],
    requested_format: str,
    max_report: int,
) -> SplitAudit:
    stats = SplitAudit(split=split)
    format_used: str | None = None
    target = np.arange(1, 10, dtype=np.int16)

    for start, raw_inputs, raw_labels in chunks:
        if raw_inputs.shape[0] == 0:
            continue
        if format_used is None:
            format_used = _infer_format(raw_inputs, raw_labels) if requested_format == FORMAT_AUTO else requested_format
            stats.format_used = format_used

        raw_inputs = raw_inputs.astype(np.int16, copy=False)
        raw_labels = raw_labels.astype(np.int16, copy=False)

        stats.raw_input_min = min(stats.raw_input_min, int(raw_inputs.min()))
        stats.raw_input_max = max(stats.raw_input_max, int(raw_inputs.max()))
        stats.raw_label_min = min(stats.raw_label_min, int(raw_labels.min()))
        stats.raw_label_max = max(stats.raw_label_max, int(raw_labels.max()))
        stats.raw_input_values.update(int(v) for v in np.unique(raw_inputs))
        stats.raw_label_values.update(int(v) for v in np.unique(raw_labels))

        inputs, labels = _normalize(raw_inputs, raw_labels, format_used)

        stats.norm_input_min = min(stats.norm_input_min, int(inputs.min()))
        stats.norm_input_max = max(stats.norm_input_max, int(inputs.max()))
        stats.norm_label_min = min(stats.norm_label_min, int(labels.min()))
        stats.norm_label_max = max(stats.norm_label_max, int(labels.max()))

        stats.num_examples += int(inputs.shape[0])
        blank_counts = np.sum(inputs == 0, axis=(1, 2))
        stats.blank_count_total += int(blank_counts.sum())
        stats.blank_count_min = min(stats.blank_count_min, int(blank_counts.min()))
        stats.blank_count_max = max(stats.blank_count_max, int(blank_counts.max()))

        in_range_input = (inputs >= 0) & (inputs <= 9)
        in_range_label = (labels >= 1) & (labels <= 9)
        bad_input = np.any(~in_range_input, axis=(1, 2))
        bad_label = np.any(~in_range_label, axis=(1, 2))

        stats.num_input_out_of_range_examples += int(bad_input.sum())
        stats.num_label_out_of_range_examples += int(bad_label.sum())
        _append_first_indices(stats.first_bad_indices["input_out_of_range"], bad_input, start, max_report)
        _append_first_indices(stats.first_bad_indices["label_out_of_range"], bad_label, start, max_report)

        clue_mismatch = np.any((inputs != 0) & (inputs != labels), axis=(1, 2))
        stats.num_clue_mismatch_examples += int(clue_mismatch.sum())
        _append_first_indices(stats.first_bad_indices["clue_mismatch"], clue_mismatch, start, max_report)

        row_ok = np.all(np.sort(labels, axis=2) == target, axis=2).all(axis=1)
        col_ok = np.all(np.sort(np.transpose(labels, (0, 2, 1)), axis=2) == target, axis=2).all(axis=1)
        boxes = labels.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
        box_ok = np.all(np.sort(boxes, axis=2) == target, axis=2).all(axis=1)
        solution_ok = row_ok & col_ok & box_ok & (~bad_label)

        invalid_row = ~row_ok
        invalid_col = ~col_ok
        invalid_box = ~box_ok
        invalid_solution = ~solution_ok

        stats.num_invalid_row_examples += int(invalid_row.sum())
        stats.num_invalid_col_examples += int(invalid_col.sum())
        stats.num_invalid_box_examples += int(invalid_box.sum())
        stats.num_invalid_solution_examples += int(invalid_solution.sum())
        _append_first_indices(stats.first_bad_indices["invalid_row"], invalid_row, start, max_report)
        _append_first_indices(stats.first_bad_indices["invalid_col"], invalid_col, start, max_report)
        _append_first_indices(stats.first_bad_indices["invalid_box"], invalid_box, start, max_report)
        _append_first_indices(stats.first_bad_indices["invalid_solution"], invalid_solution, start, max_report)

        # Order-sensitive fingerprint over normalized values.
        stats.digest_sha256.update(np.ascontiguousarray(inputs, dtype=np.uint8).tobytes())
        stats.digest_sha256.update(np.ascontiguousarray(labels, dtype=np.uint8).tobytes())

    if stats.format_used is None:
        stats.format_used = requested_format if requested_format != FORMAT_AUTO else FORMAT_RAW

    return stats


def _compare_hf_tiny_ordered(
    hf_split: Dataset,
    tiny_dir: str,
    split: str,
    chunk_size: int,
    hf_format: str,
    tiny_format: str,
    max_report: int,
) -> dict:
    total_hf = len(hf_split)
    tiny_total, tiny_chunks = _iter_tiny_chunks(tiny_dir, split, chunk_size)
    if total_hf != tiny_total:
        return {
            "split": split,
            "comparable": False,
            "reason": f"count_mismatch hf={total_hf} tiny={tiny_total}",
        }

    tiny_iter = iter(tiny_chunks)
    mismatch_examples = 0
    first_mismatch_indices: list[int] = []

    for hf_start, hf_in_raw, hf_lb_raw in _iter_hf_chunks(hf_split, chunk_size):
        tiny_start, tiny_in_raw, tiny_lb_raw = next(tiny_iter)
        if hf_start != tiny_start:
            raise RuntimeError(f"Iterator offset mismatch hf={hf_start}, tiny={tiny_start}")

        hf_in, hf_lb = _normalize(hf_in_raw.astype(np.int16, copy=False), hf_lb_raw.astype(np.int16, copy=False), hf_format)
        tiny_in, tiny_lb = _normalize(
            tiny_in_raw.astype(np.int16, copy=False), tiny_lb_raw.astype(np.int16, copy=False), tiny_format
        )

        diff_mask = np.any(hf_in != tiny_in, axis=(1, 2)) | np.any(hf_lb != tiny_lb, axis=(1, 2))
        mismatch_examples += int(diff_mask.sum())
        _append_first_indices(first_mismatch_indices, diff_mask, hf_start, max_report)

    return {
        "split": split,
        "comparable": True,
        "num_examples": total_hf,
        "mismatch_examples": mismatch_examples,
        "exact_match_ordered_after_normalization": mismatch_examples == 0,
        "first_mismatch_indices": first_mismatch_indices,
    }


def _pair_hashes_from_hf(
    hf_split: Dataset,
    fmt: str,
    chunk_size: int,
    weights1: np.ndarray,
    weights2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = len(hf_split)
    h1 = np.empty(total, dtype=np.uint64)
    h2 = np.empty(total, dtype=np.uint64)
    for start, in_raw, lb_raw in _iter_hf_chunks(hf_split, chunk_size):
        end = start + in_raw.shape[0]
        in_norm, lb_norm = _normalize(
            in_raw.astype(np.int16, copy=False),
            lb_raw.astype(np.int16, copy=False),
            fmt,
        )
        flat = np.concatenate([in_norm.reshape(-1, 81), lb_norm.reshape(-1, 81)], axis=1).astype(np.uint64, copy=False)
        h1[start:end] = (flat * weights1).sum(axis=1, dtype=np.uint64)
        h2[start:end] = (flat * weights2).sum(axis=1, dtype=np.uint64)
    return h1, h2


def _pair_hashes_from_tiny(
    tiny_dir: str,
    split: str,
    fmt: str,
    chunk_size: int,
    weights1: np.ndarray,
    weights2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total, chunks = _iter_tiny_chunks(tiny_dir, split, chunk_size)
    h1 = np.empty(total, dtype=np.uint64)
    h2 = np.empty(total, dtype=np.uint64)
    for start, in_raw, lb_raw in chunks:
        end = start + in_raw.shape[0]
        in_norm, lb_norm = _normalize(
            in_raw.astype(np.int16, copy=False),
            lb_raw.astype(np.int16, copy=False),
            fmt,
        )
        flat = np.concatenate([in_norm.reshape(-1, 81), lb_norm.reshape(-1, 81)], axis=1).astype(np.uint64, copy=False)
        h1[start:end] = (flat * weights1).sum(axis=1, dtype=np.uint64)
        h2[start:end] = (flat * weights2).sum(axis=1, dtype=np.uint64)
    return h1, h2


def _unordered_overlap_from_hash_pairs(
    h1_a: np.ndarray,
    h2_a: np.ndarray,
    h1_b: np.ndarray,
    h2_b: np.ndarray,
) -> int:
    dtype = np.dtype([("h1", np.uint64), ("h2", np.uint64)])
    a = np.empty(h1_a.shape[0], dtype=dtype)
    b = np.empty(h1_b.shape[0], dtype=dtype)
    a["h1"], a["h2"] = h1_a, h2_a
    b["h1"], b["h2"] = h1_b, h2_b
    a.sort(order=["h1", "h2"])
    b.sort(order=["h1", "h2"])
    i = 0
    j = 0
    inter = 0
    while i < a.shape[0] and j < b.shape[0]:
        av_h1 = int(a["h1"][i])
        av_h2 = int(a["h2"][i])
        bv_h1 = int(b["h1"][j])
        bv_h2 = int(b["h2"][j])
        if av_h1 == bv_h1 and av_h2 == bv_h2:
            inter += 1
            i += 1
            j += 1
        elif (av_h1, av_h2) < (bv_h1, bv_h2):
            i += 1
        else:
            j += 1
    return inter


def _compare_hf_tiny_unordered_overlap(
    hf_split: Dataset,
    tiny_dir: str,
    split: str,
    chunk_size: int,
    hf_format: str,
    tiny_format: str,
) -> dict:
    total_hf = len(hf_split)
    tiny_total, _ = _iter_tiny_chunks(tiny_dir, split, chunk_size)
    if total_hf != tiny_total:
        return {
            "split": split,
            "comparable": False,
            "reason": f"count_mismatch hf={total_hf} tiny={tiny_total}",
        }

    rng = np.random.default_rng(12345)
    weights1 = rng.integers(1, np.iinfo(np.uint64).max, size=162, dtype=np.uint64)
    weights2 = rng.integers(1, np.iinfo(np.uint64).max, size=162, dtype=np.uint64)

    hf_h1, hf_h2 = _pair_hashes_from_hf(hf_split, hf_format, chunk_size, weights1, weights2)
    tiny_h1, tiny_h2 = _pair_hashes_from_tiny(tiny_dir, split, tiny_format, chunk_size, weights1, weights2)
    overlap = _unordered_overlap_from_hash_pairs(hf_h1, hf_h2, tiny_h1, tiny_h2)
    return {
        "split": split,
        "comparable": True,
        "num_examples": total_hf,
        "unordered_overlap_examples_hash128": overlap,
        "unordered_overlap_ratio": float(overlap) / float(total_hf) if total_hf else 0.0,
        "note": "Hash-based 128-bit overlap (very low collision probability, not cryptographic proof).",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Sudoku datasets and compare HF vs Tiny formats/content.")
    parser.add_argument("--hf-dataset", type=str, default=None, help="HF dataset name or local save_to_disk path.")
    parser.add_argument("--tiny-dir", type=str, default=None, help="TinyRecursiveModels sudoku dataset directory.")
    parser.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits to audit.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Chunk size for iteration.")
    parser.add_argument("--hf-format", choices=ALLOWED_FORMATS, default=FORMAT_AUTO, help="HF value format.")
    parser.add_argument("--tiny-format", choices=ALLOWED_FORMATS, default=FORMAT_AUTO, help="Tiny value format.")
    parser.add_argument("--max-report", type=int, default=5, help="Max example indices to store per error type.")
    parser.add_argument(
        "--skip-ordered-compare",
        action="store_true",
        help="Skip expensive exact ordered equality check between HF and Tiny.",
    )
    parser.add_argument(
        "--compute-unordered-overlap",
        action="store_true",
        help="Compute order-insensitive overlap between HF and Tiny using 128-bit pair hashes.",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits specified.")
    if args.hf_dataset is None and args.tiny_dir is None:
        raise ValueError("Provide at least one of --hf-dataset or --tiny-dir.")

    result: dict[str, object] = {"splits": splits}

    hf_ds: DatasetDict | None = None
    if args.hf_dataset is not None:
        hf_ds = _load_hf_dataset(args.hf_dataset)
        hf_report: dict[str, object] = {}
        for split in splits:
            if split not in hf_ds:
                hf_report[split] = {"error": f"missing split '{split}'"}
                continue
            audit = _audit_chunks(
                split=split,
                chunks=_iter_hf_chunks(hf_ds[split], args.chunk_size),
                requested_format=args.hf_format,
                max_report=args.max_report,
            )
            hf_report[split] = audit.to_json()
        result["hf"] = {"dataset": args.hf_dataset, "report": hf_report}

    if args.tiny_dir is not None:
        tiny_report: dict[str, object] = {}
        for split in splits:
            try:
                _total, chunks = _iter_tiny_chunks(args.tiny_dir, split, args.chunk_size)
                audit = _audit_chunks(
                    split=split,
                    chunks=chunks,
                    requested_format=args.tiny_format,
                    max_report=args.max_report,
                )
                tiny_report[split] = audit.to_json()
            except Exception as exc:  # pylint: disable=broad-except
                tiny_report[split] = {"error": str(exc)}
        result["tiny"] = {"dataset_dir": args.tiny_dir, "report": tiny_report}

    if hf_ds is not None and args.tiny_dir is not None and not args.skip_ordered_compare:
        compare_report: dict[str, object] = {}
        hf_reports = result["hf"]["report"]  # type: ignore[index]
        tiny_reports = result["tiny"]["report"]  # type: ignore[index]
        for split in splits:
            hf_split_report = hf_reports.get(split) if isinstance(hf_reports, dict) else None
            tiny_split_report = tiny_reports.get(split) if isinstance(tiny_reports, dict) else None
            if not isinstance(hf_split_report, dict) or "error" in hf_split_report:
                compare_report[split] = {"comparable": False, "reason": "hf_split_missing_or_error"}
                continue
            if not isinstance(tiny_split_report, dict) or "error" in tiny_split_report:
                compare_report[split] = {"comparable": False, "reason": "tiny_split_missing_or_error"}
                continue
            hf_fmt = str(hf_split_report["format_used"])
            tiny_fmt = str(tiny_split_report["format_used"])
            compare_report[split] = _compare_hf_tiny_ordered(
                hf_split=hf_ds[split],
                tiny_dir=args.tiny_dir,
                split=split,
                chunk_size=args.chunk_size,
                hf_format=hf_fmt,
                tiny_format=tiny_fmt,
                max_report=args.max_report,
            )
        result["hf_vs_tiny_ordered"] = compare_report

    if hf_ds is not None and args.tiny_dir is not None and args.compute_unordered_overlap:
        compare_report: dict[str, object] = {}
        hf_reports = result["hf"]["report"]  # type: ignore[index]
        tiny_reports = result["tiny"]["report"]  # type: ignore[index]
        for split in splits:
            hf_split_report = hf_reports.get(split) if isinstance(hf_reports, dict) else None
            tiny_split_report = tiny_reports.get(split) if isinstance(tiny_reports, dict) else None
            if not isinstance(hf_split_report, dict) or "error" in hf_split_report:
                compare_report[split] = {"comparable": False, "reason": "hf_split_missing_or_error"}
                continue
            if not isinstance(tiny_split_report, dict) or "error" in tiny_split_report:
                compare_report[split] = {"comparable": False, "reason": "tiny_split_missing_or_error"}
                continue
            hf_fmt = str(hf_split_report["format_used"])
            tiny_fmt = str(tiny_split_report["format_used"])
            compare_report[split] = _compare_hf_tiny_unordered_overlap(
                hf_split=hf_ds[split],
                tiny_dir=args.tiny_dir,
                split=split,
                chunk_size=args.chunk_size,
                hf_format=hf_fmt,
                tiny_format=tiny_fmt,
            )
        result["hf_vs_tiny_unordered_overlap"] = compare_report

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
