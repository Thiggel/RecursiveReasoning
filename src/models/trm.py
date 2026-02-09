import torch
import torch.nn as nn

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
        std = 1.0
        h_init = torch.empty(config.d_model)
        l_init = torch.empty(config.d_model)
        nn.init.trunc_normal_(h_init, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(l_init, mean=0.0, std=std, a=-2 * std, b=2 * std)
        self.register_buffer("h_init", h_init)
        self.register_buffer("l_init", l_init)
        self.post_init()

    def _init_state(self, x: torch.Tensor) -> RecurrenceState:
        h = self.h_init.to(dtype=x.dtype, device=x.device)[None, None, :]
        l = self.l_init.to(dtype=x.dtype, device=x.device)[None, None, :]
        slow_state = h.expand_as(x)
        fast_state = l.expand_as(x)
        return RecurrenceState(slow=slow_state, fast=fast_state)

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
        state = self._init_state(x)
        aux_loss = x.new_zeros(())
        step_losses: list[torch.Tensor] = []

        def _act_step_loss(step_state: RecurrenceState, _step_idx: int) -> None:
            self._append_step_loss(
                step_losses=step_losses,
                hidden=step_state.slow,
                labels=labels,
                aux_loss=aux_loss,
            )

        state = self.run_act_slow_fast(
            state,
            act_steps=self.config.act_steps,
            slow_cycles=self.config.slow_steps,
            fast_cycles=self.config.fast_steps,
            fast_step_fn=lambda s: self._fast_step(s, x, attention_mask),
            slow_step_fn=lambda s: self._slow_step(s, attention_mask),
            act_step_end_fn=_act_step_loss if self.training and labels is not None else None,
            training=self.training,
            tbptt_steps=self.config.tbptt_steps,
        )
        return self._finalize(state.slow, labels, aux_loss, step_losses=step_losses)
