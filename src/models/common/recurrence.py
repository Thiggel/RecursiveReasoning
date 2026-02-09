from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import torch

from .utils import run_steps, run_act_steps


@dataclass(frozen=True)
class RecurrenceState:
    slow: torch.Tensor
    fast: torch.Tensor | None = None


class RecurrenceMixin:
    @staticmethod
    def run_steps(*args, **kwargs):
        return run_steps(*args, **kwargs)

    @staticmethod
    def run_act_steps(*args, **kwargs):
        return run_act_steps(*args, **kwargs)

    @staticmethod
    def run_act_slow_fast(
        state: RecurrenceState,
        *,
        act_steps: int,
        slow_cycles: int,
        fast_cycles: int,
        fast_step_fn,
        slow_step_fn=None,
        act_step_end_fn=None,
        training: bool,
        tbptt_steps: int,
    ) -> RecurrenceState:
        act_steps = int(act_steps)
        if act_steps <= 0:
            return state
        detach_period = None
        if training:
            detach_period = int(tbptt_steps) if int(tbptt_steps) > 0 else 1
        for act_step in range(act_steps):
            # Within an ACT step: run slow_cycles; only the final slow cycle keeps grad.
            for slow_step in range(int(slow_cycles)):
                no_grad = slow_step < int(slow_cycles) - 1
                ctx = torch.no_grad() if no_grad else nullcontext()
                with ctx:
                    for _ in range(int(fast_cycles)):
                        state = fast_step_fn(state)
                    if slow_step_fn is not None:
                        state = slow_step_fn(state)
            if act_step_end_fn is not None:
                act_step_end_fn(state, act_step)
            # Detach carry according to TBPTT window (matching TinyRecursiveModels semantics)
            if detach_period is not None and (act_step + 1) % detach_period == 0:
                state = RecurrenceState(
                    slow=state.slow.detach(),
                    fast=state.fast.detach() if state.fast is not None else None,
                )
            elif not training:
                # Always detach in eval to avoid graph accumulation
                state = RecurrenceState(
                    slow=state.slow.detach(),
                    fast=state.fast.detach() if state.fast is not None else None,
                )
        return state

    @staticmethod
    def run_single_state(
        *,
        state: RecurrenceState,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_stack,
        cache_enabled: bool,
        legacy_past: list | None,
        new_past: list,
        inject_fn,
        slow_steps: int,
        fast_steps: int,
        act_steps: int,
        act_step_end_fn=None,
        training: bool,
        tbptt_steps: int,
    ) -> RecurrenceState:
        cache_step = 0

        def _fast_step(inner_state: RecurrenceState) -> RecurrenceState:
            nonlocal cache_step
            hidden = inject_fn(input_tensor, inner_state.slow)
            if cache_enabled:
                start = cache_step * shared_stack.num_layers
                end = start + shared_stack.num_layers
                past_slice = legacy_past[start:end] if legacy_past is not None else None
                hidden, present = shared_stack(hidden, attention_mask, past_key_values=past_slice, use_cache=True)
                new_past.extend(present)
            else:
                hidden, _ = shared_stack(hidden, attention_mask)
            cache_step += 1
            return RecurrenceState(slow=hidden, fast=None)

        return RecurrenceMixin.run_act_slow_fast(
            state,
            act_steps=act_steps,
            slow_cycles=slow_steps,
            fast_cycles=fast_steps,
            fast_step_fn=_fast_step,
            slow_step_fn=None,
            act_step_end_fn=act_step_end_fn,
            training=training,
            tbptt_steps=tbptt_steps,
        )
