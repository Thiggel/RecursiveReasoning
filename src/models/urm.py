import torch
from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, TransformerConfig, detach_state, prepare_kv_cache


class URMModel(BaseModel):
    """
    Shared block with optional ACT over per-token halting.
    If ACT is disabled, runs a fixed number of loops.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.shared = BlockStack(config, num_layers=config.n_layers)
        self.post_init()

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
        cache_enabled, legacy_past, new_past = prepare_kv_cache(
            past_key_values, use_cache=use_cache, causal=self.config.causal,
        )

        h = x
        loops = max(1, int(self.config.loops))
        tbptt_steps = int(self.config.tbptt_steps)
        for step in range(loops):
            if cache_enabled:
                start = step * self.shared.num_layers
                end = start + self.shared.num_layers
                past_slice = legacy_past[start:end] if legacy_past is not None else None
                h, present = self.shared(h, attention_mask, past_key_values=past_slice, use_cache=True)
                new_past.extend(present)
            else:
                h, _ = self.shared(h, attention_mask)
            if self.training and tbptt_steps > 0 and (step + 1) % tbptt_steps == 0:
                h = detach_state(h)
        return self._finalize(h, labels, x.new_zeros(()), past_key_values=new_past)
