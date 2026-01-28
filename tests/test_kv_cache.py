import types

import torch

from src.models.common.config import TransformerConfig
from src.models.drm import DRMModel
from src.models.urm import URMModel


def _make_cache_items(count: int):
    items = []
    for i in range(count):
        val = torch.full((1, 1, 1, 1), float(i))
        items.append((val, val.clone()))
    return items


def _expected_indices(step: int, num_layers: int):
    start = step * num_layers
    end = start + num_layers
    return list(range(start, end))


def _extract_legacy_cache(cache):
    if cache is None:
        return None
    if isinstance(cache, list):
        return cache
    if hasattr(cache, "layers"):
        return [(k, v) for k, v, _ in cache]
    raise TypeError(f"Unknown cache type: {type(cache)}")


def test_urm_cache_slice_order():
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        vocab_size=11,
        use_puzzle_emb=False,
        causal=True,
        loops=3,
    )
    model = URMModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    past_key_values = _make_cache_items(cfg.n_layers * cfg.loops)

    records = []

    def fake_forward(self, x, attention_mask, past_key_values=None, use_cache=False):
        records.append(past_key_values)
        present = None
        if use_cache:
            present = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)) for _ in range(self.num_layers)]
        return x, present

    model.shared.forward = types.MethodType(fake_forward, model.shared)
    model(input_ids, use_cache=True, past_key_values=past_key_values)

    assert len(records) == cfg.loops
    for step, slice_kv in enumerate(records):
        expected = _expected_indices(step, cfg.n_layers)
        assert slice_kv is not None
        assert len(slice_kv) == cfg.n_layers
        for idx, (k, v) in zip(expected, slice_kv):
            assert int(k.item()) == idx
            assert int(v.item()) == idx


def test_drm_cache_slice_order():
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        vocab_size=11,
        use_puzzle_emb=False,
        causal=True,
        loops=3,
        state_init="zero",
    )
    model = DRMModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    past_key_values = _make_cache_items(cfg.n_layers * cfg.loops)

    records = []

    def fake_forward(self, x, attention_mask, past_key_values=None, use_cache=False):
        records.append(past_key_values)
        present = None
        if use_cache:
            present = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)) for _ in range(self.num_layers)]
        return x, present

    model.shared.forward = types.MethodType(fake_forward, model.shared)
    model(input_ids, use_cache=True, past_key_values=past_key_values)

    assert len(records) == cfg.loops
    for step, slice_kv in enumerate(records):
        expected = _expected_indices(step, cfg.n_layers)
        assert slice_kv is not None
        assert len(slice_kv) == cfg.n_layers
        for idx, (k, v) in zip(expected, slice_kv):
            assert int(k.item()) == idx
            assert int(v.item()) == idx


def test_urm_cache_length():
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        vocab_size=11,
        use_puzzle_emb=False,
        causal=True,
        loops=3,
    )
    model = URMModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    outputs = model(input_ids, use_cache=True)
    legacy = _extract_legacy_cache(outputs.past_key_values)
    assert len(legacy) == cfg.loops * cfg.n_layers
