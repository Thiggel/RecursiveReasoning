import torch

from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, TransformerConfig, init_state_like, detach_state, prepare_kv_cache


class DRMModel(BaseModel):
    """
    Decoder with noise-initialized recurrent state.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.shared = BlockStack(config, num_layers=config.n_layers)
        self.state_proj = torch.nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.post_init()

    def _step(self, x: torch.Tensor, s_prev: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        h_in = self.state_proj(torch.cat([x, s_prev], dim=-1))
        h, _ = self.shared(h_in, attention_mask)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        puzzle_identifiers: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutput:
        x = self.embed(input_ids, puzzle_identifiers)
        s0 = init_state_like(x, self.config.state_init, self.config.state_sigma)

        cache_enabled, legacy_past, new_past = prepare_kv_cache(
            past_key_values, use_cache=use_cache, causal=self.config.causal,
        )

        s = s0
        tbptt_steps = int(self.config.tbptt_steps)
        for step in range(int(self.config.loops)):
            if cache_enabled:
                start = step * self.shared.num_layers
                end = start + self.shared.num_layers
                past_slice = legacy_past[start:end] if legacy_past is not None else None
                h_in = self.state_proj(torch.cat([x, s], dim=-1))
                s, present = self.shared(h_in, attention_mask, past_key_values=past_slice, use_cache=True)
                new_past.extend(present)
            else:
                s = self._step(x, s, attention_mask)
            if self.training and tbptt_steps > 0 and (step + 1) % tbptt_steps == 0:
                s = detach_state(s)
        h = s
        return self._finalize(h, labels, x.new_zeros(()), past_key_values=new_past)
