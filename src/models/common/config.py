from __future__ import annotations

from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    """
    Shared config for all model variants.
    Accepts nested dictionaries for hrm/trm/state for minimal caller code.
    """
    model_type = "minimal_postnorm_rope_transformer"

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
        loops: int = 8,
        hrm_fast_loops: int = 4,
        hrm_slow_loops: int = 2,
        hrm_fast_layers: int | None = None,
        hrm_slow_layers: int | None = None,
        use_puzzle_emb: bool = False,
        num_puzzle_ids: int = 1,
        state_init: str = "zero",
        state_sigma: float = 0.02,
        trm_init_y_from_x: bool = True,
        recurrence: str = "none",
        hrm: dict | None = None,
        trm: dict | None = None,
        state: dict | None = None,
        tbptt_steps: int = 1,
        use_convswiglu: bool = False,
        convswiglu_kernel_size: int = 3,
        convswiglu_groups: int | None = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        if hrm:
            hrm_fast_loops = hrm.get("fast_loops", hrm_fast_loops)
            hrm_slow_loops = hrm.get("slow_loops", hrm_slow_loops)
            hrm_fast_layers = hrm.get("fast_layers", hrm_fast_layers)
            hrm_slow_layers = hrm.get("slow_layers", hrm_slow_layers)

        if trm:
            trm_init_y_from_x = trm.get("init_y_from_x", trm_init_y_from_x)

        if state:
            state_init = state.get("init", state_init)
            state_sigma = state.get("sigma", state_sigma)

        self.vocab_size = vocab_size
        self.sep_token_id = sep_token_id

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.rope_theta = rope_theta
        self.causal = causal

        self.n_layers = n_layers
        self.loops = loops
        self.hrm_fast_loops = hrm_fast_loops
        self.hrm_slow_loops = hrm_slow_loops
        self.hrm_fast_layers = hrm_fast_layers if hrm_fast_layers is not None else n_layers
        self.hrm_slow_layers = hrm_slow_layers if hrm_slow_layers is not None else n_layers

        self.use_puzzle_emb = use_puzzle_emb
        self.num_puzzle_ids = num_puzzle_ids

        self.state_init = state_init
        self.state_sigma = state_sigma

        self.trm_init_y_from_x = trm_init_y_from_x
        self.recurrence = recurrence

        self.is_decoder = bool(causal)

        self.tbptt_steps = tbptt_steps
        self.use_convswiglu = use_convswiglu
        self.convswiglu_kernel_size = convswiglu_kernel_size
        self.convswiglu_groups = convswiglu_groups
