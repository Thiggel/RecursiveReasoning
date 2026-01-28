import torch

from transformers.modeling_outputs import CausalLMOutput

from .common import BaseModel, BlockStack, TransformerConfig, detach_state


class TRMModel(BaseModel):
    """
    Two coupled per-token states (y, z) with shared block updates.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.shared = BlockStack(config, num_layers=config.n_layers)
        self.y_proj = torch.nn.Linear(3 * config.d_model, config.d_model, bias=False)
        self.z_proj = torch.nn.Linear(3 * config.d_model, config.d_model, bias=False)
        self.post_init()

    def _step(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None):
        y, z = state
        cat = torch.cat([x, y, z], dim=-1)
        y_in = self.y_proj(cat)
        z_in = self.z_proj(cat)
        y_next, _ = self.shared(y_in, attention_mask)
        z_next, _ = self.shared(z_in, attention_mask)
        return (y_next, z_next)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        puzzle_identifiers: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        x = self.embed(input_ids, puzzle_identifiers)

        y0 = x.clone() if self.config.trm_init_y_from_x else torch.zeros_like(x)
        z0 = torch.zeros_like(x)

        y, z = y0, z0
        tbptt_steps = int(self.config.tbptt_steps)
        for step in range(int(self.config.loops)):
            y, z = self._step(x, (y, z), attention_mask)
            if self.training and tbptt_steps > 0 and (step + 1) % tbptt_steps == 0:
                y, z = detach_state((y, z))
        h = y
        return self._finalize(h, labels, x.new_zeros(()))
