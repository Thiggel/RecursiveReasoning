import torch

from src.models.common.recurrence import RecurrenceMixin, RecurrenceState
from src.models.trm import TRMModel
from src.models.hrm import HRMModel
from src.models.urm import URMModel
from src.models.drm import DRMModel
from src.models.common.config import TRMConfig, HRMConfig, URMConfig, DRMConfig


def _identity_forward(self, x, attention_mask, past_key_values=None, use_cache=False):
    present = None
    if use_cache:
        present = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)) for _ in range(self.num_layers)]
    return x, present


def test_run_act_slow_fast_counts():
    state = RecurrenceState(slow=torch.zeros(1), fast=torch.zeros(1))
    counts = {"fast": 0, "slow": 0}

    def fast_step(s):
        counts["fast"] += 1
        return s

    def slow_step(s):
        counts["slow"] += 1
        return s

    RecurrenceMixin.run_act_slow_fast(
        state,
        act_steps=2,
        slow_cycles=3,
        fast_cycles=4,
        fast_step_fn=fast_step,
        slow_step_fn=slow_step,
        training=True,
        tbptt_steps=1,
    )
    assert counts["fast"] == 24
    assert counts["slow"] == 6


def test_trm_fast_slow_injection():
    cfg = TRMConfig(
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        trm_l_inject=True,
        slow_steps=1,
        fast_steps=1,
        act_steps=1,
    )
    model = TRMModel(cfg)
    model.shared.forward = _identity_forward.__get__(model.shared, type(model.shared))

    x = torch.ones(1, 2, cfg.d_model)
    state = RecurrenceState(slow=x.clone(), fast=torch.zeros_like(x))
    state = model._fast_step(state, x, None)
    assert torch.allclose(state.fast, 2 * x)

    state = model._slow_step(state, None)
    assert torch.allclose(state.slow, 3 * x)


def test_hrm_fast_slow_injection():
    cfg = HRMConfig(
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        hrm_fast_layers=1,
        hrm_slow_layers=1,
        hrm_l_inject=True,
        slow_steps=1,
        fast_steps=1,
        act_steps=1,
    )
    model = HRMModel(cfg)
    model.fast.forward = _identity_forward.__get__(model.fast, type(model.fast))
    model.slow.forward = _identity_forward.__get__(model.slow, type(model.slow))

    x = torch.ones(1, 2, cfg.d_model)
    state = RecurrenceState(slow=x.clone(), fast=torch.zeros_like(x))
    state = model._fast_step(state, x, None)
    assert torch.allclose(state.fast, 2 * x)

    state = model._slow_step(state, None)
    assert torch.allclose(state.slow, 3 * x)


def test_urm_single_state_injection_add():
    cfg = URMConfig(
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        causal=False,
        slow_steps=1,
        fast_steps=1,
        act_steps=1,
    )
    model = URMModel(cfg)
    model.shared.forward = _identity_forward.__get__(model.shared, type(model.shared))

    x = torch.ones(1, 2, cfg.d_model)
    state = RecurrenceState(slow=x.clone(), fast=None)
    state = model.run_single_state(
        state=state,
        input_tensor=x,
        attention_mask=None,
        shared_stack=model.shared,
        cache_enabled=False,
        legacy_past=None,
        new_past=[],
        inject_fn=lambda inp, hidden: hidden + inp,
        slow_steps=1,
        fast_steps=1,
        act_steps=1,
        training=True,
        tbptt_steps=0,
    )
    assert torch.allclose(state.slow, 2 * x)


def test_drm_inject_add_avoids_projection():
    cfg = DRMConfig(
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        causal=False,
        slow_steps=1,
        fast_steps=1,
        act_steps=1,
        state_init="zero",
        drm_inject="add",
    )
    model = DRMModel(cfg)
    called = {"count": 0}

    def _proj_forward(self, x):
        called["count"] += 1
        return x

    model.state_proj.forward = _proj_forward.__get__(model.state_proj, type(model.state_proj))
    x = torch.ones(1, 2, cfg.d_model)
    hidden = torch.zeros_like(x)
    out = model._inject(x, hidden)
    assert called["count"] == 0
    assert torch.allclose(out, x)


def test_drm_inject_concat_uses_projection():
    cfg = DRMConfig(
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        causal=False,
        slow_steps=1,
        fast_steps=1,
        act_steps=1,
        state_init="zero",
        drm_inject="concat",
    )
    model = DRMModel(cfg)
    called = {"count": 0}

    def _proj_forward(self, x):
        called["count"] += 1
        return x[:, :, : cfg.d_model]

    model.state_proj.forward = _proj_forward.__get__(model.state_proj, type(model.state_proj))
    x = torch.ones(1, 2, cfg.d_model)
    hidden = torch.zeros_like(x)
    _ = model._inject(x, hidden)
    assert called["count"] == 1
