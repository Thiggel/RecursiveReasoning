from __future__ import annotations

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import BaseTransformerConfig
from .embeddings import TokenAndPuzzleEmbedding
from .utils import compute_loss


class BaseModel(PreTrainedModel, GenerationMixin):
    config_class = BaseTransformerConfig

    def __init__(self, config: BaseTransformerConfig):
        super().__init__(config)
        self.embed = TokenAndPuzzleEmbedding(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed.tok

    def _compute_logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(h)
        if getattr(self.config, "loss_type", "cross_entropy") == "stablemax":
            logits.use_stablemax = True
        return logits

    def _append_step_loss(
        self,
        *,
        step_losses: list[torch.Tensor],
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        aux_loss: torch.Tensor,
    ) -> None:
        if labels is None:
            return
        step_logits = self._compute_logits(hidden)
        step_loss = compute_loss(step_logits, labels, aux_loss)
        if step_loss is not None:
            step_losses.append(step_loss)

    def _finalize(
        self,
        h: torch.Tensor,
        labels: torch.Tensor | None,
        aux_loss: torch.Tensor,
        past_key_values: list | None = None,
        step_losses: list[torch.Tensor] | None = None,
    ) -> CausalLMOutputWithPast:
        logits = self._compute_logits(h)
        loss = compute_loss(logits, labels, aux_loss)
        if labels is not None and step_losses:
            loss = torch.stack(step_losses).mean()
        cache = None
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache = past_key_values
            else:
                cache = DynamicCache(ddp_cache_data=past_key_values)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)
