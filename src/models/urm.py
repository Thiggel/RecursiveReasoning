import torch
from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, URMConfig, RecurrenceMixin, RecurrenceState, prepare_kv_cache


class URMModel(BaseModel, RecurrenceMixin):
    """
    Shared block with a single recurrent state.
    """
    config_class = URMConfig

    def __init__(self, config: URMConfig):
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
        aux_loss = x.new_zeros(())
        step_losses: list[torch.Tensor] = []
        cache_enabled, legacy_past, new_past = prepare_kv_cache(
            past_key_values, use_cache=use_cache, causal=self.config.causal,
        )

        def _act_step_loss(step_state: RecurrenceState, _step_idx: int) -> None:
            self._append_step_loss(
                step_losses=step_losses,
                hidden=step_state.slow,
                labels=labels,
                aux_loss=aux_loss,
            )

        state = RecurrenceState(slow=x, fast=None)
        state = self.run_single_state(
            state=state,
            input_tensor=x,
            attention_mask=attention_mask,
            shared_stack=self.shared,
            cache_enabled=cache_enabled,
            legacy_past=legacy_past,
            new_past=new_past,
            inject_fn=lambda inp, hidden: hidden + inp,
            slow_steps=self.config.slow_steps,
            fast_steps=self.config.fast_steps,
            act_steps=self.config.act_steps,
            act_step_end_fn=_act_step_loss if self.training and labels is not None else None,
            training=self.training,
            tbptt_steps=self.config.tbptt_steps,
        )
        return self._finalize(state.slow, labels, aux_loss, past_key_values=new_past, step_losses=step_losses)
