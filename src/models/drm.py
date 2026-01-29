import torch

from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, DRMConfig, RecurrenceMixin, RecurrenceState, init_state_like, prepare_kv_cache


class DRMModel(BaseModel, RecurrenceMixin):
    """
    Decoder with noise-initialized recurrent state.
    """
    config_class = DRMConfig

    def __init__(self, config: DRMConfig):
        super().__init__(config)
        self.shared = BlockStack(config, num_layers=config.n_layers)
        self.state_proj = torch.nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.post_init()

    def _inject(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if self.config.drm_inject == "add":
            return hidden + x
        return self.state_proj(torch.cat([x, hidden], dim=-1))

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

        state = RecurrenceState(slow=s0, fast=None)
        state = self.run_single_state(
            state=state,
            input_tensor=x,
            attention_mask=attention_mask,
            shared_stack=self.shared,
            cache_enabled=cache_enabled,
            legacy_past=legacy_past,
            new_past=new_past,
            inject_fn=self._inject,
            slow_steps=self.config.slow_steps,
            fast_steps=self.config.fast_steps,
            act_steps=self.config.act_steps,
            training=self.training,
            tbptt_steps=self.config.tbptt_steps,
        )
        h = state.slow
        return self._finalize(h, labels, x.new_zeros(()), past_key_values=new_past)
