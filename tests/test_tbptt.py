import torch

from src.models.common.config import TRMConfig, URMConfig, DRMConfig, HRMConfig
from src.models.drm import DRMModel
from src.models.hrm import HRMModel
from src.models.trm import TRMModel
from src.models.urm import URMModel


def _make_input(batch=2, seq=4, vocab=11):
    return torch.randint(0, vocab, (batch, seq))


def test_trm_tbptt_runs_with_windowed_grad():
    cfg = TRMConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
        slow_steps=2,
        fast_steps=2,
        act_steps=3,
    )
    model = TRMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert True


def test_trm_tbptt_disabled_in_eval():
    cfg = TRMConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
        slow_steps=2,
        fast_steps=2,
        act_steps=3,
    )
    model = TRMModel(cfg)
    model.eval()
    input_ids = _make_input()
    model(input_ids)
    assert True


def test_urm_tbptt_runs_with_windowed_grad():
    cfg = URMConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        loops=5,
        tbptt_steps=2,
        slow_steps=2,
        fast_steps=2,
        act_steps=5,
    )
    model = URMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert True


def test_drm_tbptt_runs_with_windowed_grad():
    cfg = DRMConfig(
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
    assert True


def test_hrm_tbptt_runs_with_windowed_grad():
    cfg = HRMConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        hrm_fast_layers=1,
        hrm_slow_layers=1,
        slow_steps=5,
        fast_steps=2,
        tbptt_steps=2,
        act_steps=3,
    )
    model = HRMModel(cfg)
    model.train()
    input_ids = _make_input()
    model(input_ids)
    assert True
