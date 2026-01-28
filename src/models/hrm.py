import torch

from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, TransformerConfig, detach_state


class HRMModel(BaseModel):
    """
    Two-block HRM: slow block wraps multiple fast block steps.
    ACT (if enabled) decides number of slow iterations.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.fast = BlockStack(config, num_layers=config.hrm_fast_layers)
        self.slow = BlockStack(config, num_layers=config.hrm_slow_layers)
        self.post_init()

    def _slow_step(self, h: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        h, _ = self.slow(h, attention_mask)
        for _ in range(int(self.config.hrm_fast_loops)):
            h, _ = self.fast(h, attention_mask)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        puzzle_identifiers: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        x = self.embed(input_ids, puzzle_identifiers)
        h = x
        tbptt_steps = int(self.config.tbptt_steps)
        for step in range(int(self.config.hrm_slow_loops)):
            h = self._slow_step(h, attention_mask)
            if self.training and tbptt_steps > 0 and (step + 1) % tbptt_steps == 0:
                h = detach_state(h)
        return self._finalize(h, labels, x.new_zeros(()))
