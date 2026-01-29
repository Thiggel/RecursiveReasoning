import torch

from src.models.common.config import URMConfig, DRMConfig
from src.models.drm import DRMModel
from src.models.urm import URMModel


def _make_config():
    return URMConfig(
        d_model=16,
        n_heads=4,
        d_ff=32,
        n_layers=2,
        vocab_size=23,
        use_puzzle_emb=False,
        causal=True,
        act_steps=2,
        tbptt_steps=1,
    )


def test_urm_generate_runs():
    cfg = _make_config()
    model = URMModel(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False, use_cache=True)
    assert out.shape == (1, 8)


def test_drm_generate_runs():
    cfg = DRMConfig(**_make_config().to_dict(), state_init="zero")
    model = DRMModel(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False, use_cache=True)
    assert out.shape == (1, 8)
