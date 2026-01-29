import torch

from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, TRMConfig, RecurrenceMixin, RecurrenceState


class TRMModel(BaseModel, RecurrenceMixin):
    """
    Two coupled per-token states (slow, fast) with shared block updates.
    """
    config_class = TRMConfig

    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.shared = BlockStack(config, num_layers=config.n_layers)
        self.post_init()

    def _fast_step(self, state: RecurrenceState, x: torch.Tensor, attention_mask: torch.Tensor | None) -> RecurrenceState:
        slow_state = state.slow
        fast_state = state.fast if state.fast is not None else state.slow
        inject = slow_state + x if self.config.trm_l_inject else slow_state
        fast_state, _ = self.shared(fast_state + inject, attention_mask)
        return RecurrenceState(slow=slow_state, fast=fast_state)

    def _slow_step(self, state: RecurrenceState, attention_mask: torch.Tensor | None) -> RecurrenceState:
        slow_state = state.slow
        fast_state = state.fast if state.fast is not None else state.slow
        slow_state, _ = self.shared(slow_state + fast_state, attention_mask)
        return RecurrenceState(slow=slow_state, fast=fast_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        puzzle_identifiers: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        x = self.embed(input_ids, puzzle_identifiers)

        slow_state = x.clone() if self.config.trm_init_y_from_x else torch.zeros_like(x)
        fast_state = torch.zeros_like(x)
        state = RecurrenceState(slow=slow_state, fast=fast_state)

        state = self.run_act_slow_fast(
            state,
            act_steps=self.config.act_steps,
            slow_cycles=self.config.slow_steps,
            fast_cycles=self.config.fast_steps,
            fast_step_fn=lambda s: self._fast_step(s, x, attention_mask),
            slow_step_fn=lambda s: self._slow_step(s, attention_mask),
            training=self.training,
            tbptt_steps=self.config.tbptt_steps,
        )
        return self._finalize(state.slow, labels, x.new_zeros(()))
