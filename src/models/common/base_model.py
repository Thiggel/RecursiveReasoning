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

    def _finalize(
        self,
        h: torch.Tensor,
        labels: torch.Tensor | None,
        aux_loss: torch.Tensor,
        past_key_values: list | None = None,
    ) -> CausalLMOutputWithPast:
        logits = self.lm_head(h)
        loss = compute_loss(logits, labels, aux_loss)
        cache = None
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache = past_key_values
            else:
                cache = DynamicCache(ddp_cache_data=past_key_values)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)
