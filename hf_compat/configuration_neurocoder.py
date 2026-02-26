"""Transformers config for NeuroCoder remote-code loading."""

from __future__ import annotations

from transformers import PretrainedConfig


class NeuroCoderConfig(PretrainedConfig):
    model_type = "neurocoder"

    def __init__(
        self,
        vocab_size: int = 32000,
        context_length: int = 4096,
        hidden_size: int = 1024,
        num_layers: int = 20,
        num_heads: int = 16,
        ffn_multiplier: int = 4,
        moe_every_n_layers: int = 2,
        num_experts: int = 8,
        router_top_k: int | None = None,
        top_k: int = 2,
        capacity_factor_train: float = 1.25,
        capacity_factor_infer: float = 1.0,
        dropout: float = 0.0,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        # Aliases expected by Transformers generation/runtime utilities.
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads
        self.max_position_embeddings = context_length
        self.use_cache = use_cache
        self.ffn_multiplier = ffn_multiplier
        self.moe_every_n_layers = moe_every_n_layers
        self.num_experts = num_experts
        # Keep MoE router top-k separate from generation top_k to avoid HF generation warnings.
        self.router_top_k = router_top_k if router_top_k is not None else top_k
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_infer = capacity_factor_infer
        self.dropout = dropout

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads
