"""Model configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TinyMoEConfig:
    vocab_size: int = 32_000
    context_length: int = 4_096
    hidden_size: int = 1_024
    num_layers: int = 20
    num_heads: int = 16
    ffn_multiplier: int = 4
    moe_every_n_layers: int = 2
    num_experts: int = 8
    top_k: int = 2
    capacity_factor_train: float = 1.25
    capacity_factor_infer: float = 1.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads
