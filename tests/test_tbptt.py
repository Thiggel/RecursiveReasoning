import torch

from src.models.common.config import TransformerConfig
from src.models.drm import DRMModel
from src.models.hrm import HRMModel
from src.models.trm import TRMModel
from src.models.urm import URMModel


def _count_detaches(monkeypatch, module):
    calls = {"count": 0}

    def _detach(state):
        calls["count"] += 1
        if isinstance(state, torch.Tensor):
            return state.detach()
        if isinstance(state, tuple):
            return tuple(s.detach() for s in state)
        return state

    monkeypatch.setattr(module, "detach_state", _detach)
    return calls


def _make_input(batch=2, seq=4, vocab=11):
    return torch.randint(0, vocab, (batch, seq))


def test_trm_tbptt_detach_count(monkeypatch):
    import src.models.trm as trm_module

    calls = _count_detaches(monkeypatch, trm_module)
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
    )
    model = TRMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert calls["count"] == 2


def test_trm_tbptt_disabled_in_eval(monkeypatch):
    import src.models.trm as trm_module

    calls = _count_detaches(monkeypatch, trm_module)
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
    )
    model = TRMModel(cfg)
    model.eval()
    input_ids = _make_input()
    model(input_ids)
    assert calls["count"] == 0


def test_urm_tbptt_detach_count(monkeypatch):
    import src.models.urm as urm_module

    calls = _count_detaches(monkeypatch, urm_module)
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
    )
    model = URMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert calls["count"] == 2


def test_drm_tbptt_detach_count(monkeypatch):
    import src.models.drm as drm_module

    calls = _count_detaches(monkeypatch, drm_module)
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
        causal=False,
        state_init="zero",
    )
    model = DRMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert calls["count"] == 2


def test_hrm_tbptt_detach_count(monkeypatch):
    import src.models.hrm as hrm_module

    calls = _count_detaches(monkeypatch, hrm_module)
    cfg = TransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        hrm_fast_layers=1,
        hrm_slow_layers=1,
        hrm_fast_loops=2,
        hrm_slow_loops=5,
        tbptt_steps=2,
    )
    model = HRMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert calls["count"] == 2
