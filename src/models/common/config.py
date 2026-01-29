from __future__ import annotations

from transformers import PretrainedConfig


class BaseTransformerConfig(PretrainedConfig):
    model_type = "minimal_transformer"

    def __init__(
        self,
        vocab_size: int = 64,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        sep_token_id: int = 2,
        eos_token_id: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        causal: bool = False,
        n_layers: int = 1,
        use_puzzle_emb: bool = False,
        num_puzzle_ids: int = 1,
        tbptt_steps: int = 1,
        use_convswiglu: bool = False,
        convswiglu_kernel_size: int = 3,
        convswiglu_groups: int | None = None,
        slow_steps: int = 1,
        fast_steps: int = 1,
        act_steps: int = 1,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.sep_token_id = sep_token_id
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.causal = causal
        self.n_layers = n_layers
        self.use_puzzle_emb = use_puzzle_emb
        self.num_puzzle_ids = num_puzzle_ids
        self.tbptt_steps = tbptt_steps
        self.use_convswiglu = use_convswiglu
        self.convswiglu_kernel_size = convswiglu_kernel_size
        self.convswiglu_groups = convswiglu_groups
        self.slow_steps = slow_steps
        self.fast_steps = fast_steps
        self.act_steps = act_steps
        self.is_decoder = bool(causal)


class TRMConfig(BaseTransformerConfig):
    model_type = "trm"

    def __init__(
        self,
        trm_init_y_from_x: bool = True,
        trm_l_inject: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trm_init_y_from_x = trm_init_y_from_x
        self.trm_l_inject = trm_l_inject


class HRMConfig(BaseTransformerConfig):
    model_type = "hrm"

    def __init__(
        self,
        hrm_fast_layers: int | None = None,
        hrm_slow_layers: int | None = None,
        hrm_l_inject: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hrm_fast_layers = hrm_fast_layers if hrm_fast_layers is not None else self.n_layers
        self.hrm_slow_layers = hrm_slow_layers if hrm_slow_layers is not None else self.n_layers
        self.hrm_l_inject = hrm_l_inject


class URMConfig(BaseTransformerConfig):
    model_type = "urm"


class DRMConfig(BaseTransformerConfig):
    model_type = "drm"

    def __init__(
        self,
        state_init: str = "zero",
        state_sigma: float = 0.02,
        drm_inject: str = "concat",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_init = state_init
        self.state_sigma = state_sigma
        self.drm_inject = drm_inject
