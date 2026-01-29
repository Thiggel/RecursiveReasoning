import torch

from src.models.common.convswiglu import ConvSwiGLU
from src.models.common.config import BaseTransformerConfig
from src.models.common.postnorm_block import PostNormBlock


def test_convswiglu_shape():
    module = ConvSwiGLU(d_model=8, d_ff=16, dropout=0.0, kernel_size=3)
    x = torch.randn(2, 5, 8)
    y = module(x)
    assert y.shape == x.shape


def test_postnorm_block_uses_convswiglu_when_enabled():
    cfg = BaseTransformerConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        vocab_size=11,
        use_puzzle_emb=False,
        use_convswiglu=True,
        convswiglu_kernel_size=3,
    )
    block = PostNormBlock(cfg)
    assert isinstance(block.mlp, ConvSwiGLU)
