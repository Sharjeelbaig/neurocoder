"""Decoder-only Transformer with sparse MoE FFN blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from model.config import TinyMoEConfig

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]


@dataclass(slots=True)
class RouterStats:
    aux_loss: float
    z_loss: float
    expert_load: list[float]
    dropped_tokens: int


if TORCH_AVAILABLE:

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x: Tensor) -> Tensor:
            rms = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(rms + self.eps)
            return x * self.weight


    class SelfAttention(nn.Module):
        def __init__(self, config: TinyMoEConfig) -> None:
            super().__init__()
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
            self.out = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, x: Tensor) -> Tensor:
            batch, seq_len, hidden = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            def shape_heads(t: Tensor) -> Tensor:
                return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            q = shape_heads(q)
            k = shape_heads(k)
            v = shape_heads(v)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, float("-inf"))
            probs = F.softmax(attn, dim=-1)
            out = torch.matmul(probs, v)
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
            return self.out(out)


    class DenseFFN(nn.Module):
        def __init__(self, config: TinyMoEConfig) -> None:
            super().__init__()
            inner = config.hidden_size * config.ffn_multiplier
            self.gate = nn.Linear(config.hidden_size, inner)
            self.up = nn.Linear(config.hidden_size, inner)
            self.down = nn.Linear(inner, config.hidden_size)

        def forward(self, x: Tensor) -> Tensor:
            return self.down(F.silu(self.gate(x)) * self.up(x))


    class MoEFeedForward(nn.Module):
        def __init__(self, config: TinyMoEConfig) -> None:
            super().__init__()
            self.num_experts = config.num_experts
            self.top_k = config.top_k
            self.capacity_factor_train = config.capacity_factor_train
            self.capacity_factor_infer = config.capacity_factor_infer
            self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.experts = nn.ModuleList([DenseFFN(config) for _ in range(config.num_experts)])

        def forward(self, x: Tensor) -> tuple[Tensor, RouterStats]:
            bsz, seq_len, hidden = x.shape
            x_flat = x.reshape(-1, hidden)
            tokens = x_flat.shape[0]

            logits = self.router(x_flat)
            probs = F.softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)

            capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_infer
            capacity = max(1, math.ceil(capacity_factor * tokens / self.num_experts))

            output = torch.zeros_like(x_flat)
            dropped_tokens = 0
            expert_load = [0.0 for _ in range(self.num_experts)]

            for expert_id in range(self.num_experts):
                expert = self.experts[expert_id]
                assigned_indices: list[Tensor] = []
                assigned_weights: list[Tensor] = []

                for rank in range(self.top_k):
                    mask = top_idx[:, rank] == expert_id
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    if idx.numel() == 0:
                        continue
                    weights = top_vals[idx, rank]
                    assigned_indices.append(idx)
                    assigned_weights.append(weights)

                if not assigned_indices:
                    continue

                token_indices = torch.cat(assigned_indices, dim=0)
                token_weights = torch.cat(assigned_weights, dim=0)

                # When capacity is exceeded, drop overflow tokens deterministically.
                if token_indices.numel() > capacity:
                    dropped_tokens += token_indices.numel() - capacity
                    token_indices = token_indices[:capacity]
                    token_weights = token_weights[:capacity]

                expert_in = x_flat[token_indices]
                expert_out = expert(expert_in)
                output[token_indices] += expert_out * token_weights.unsqueeze(-1)
                expert_load[expert_id] = float(token_indices.numel() / max(tokens, 1))

            load_tensor = torch.tensor(expert_load, device=x.device)
            mean_prob = probs.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(mean_prob * load_tensor)
            z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)

            stats = RouterStats(
                aux_loss=float(aux_loss.detach().cpu().item()),
                z_loss=float(z_loss.detach().cpu().item()),
                expert_load=expert_load,
                dropped_tokens=dropped_tokens,
            )
            return output.reshape(bsz, seq_len, hidden), stats


    class TransformerBlock(nn.Module):
        def __init__(self, config: TinyMoEConfig, use_moe: bool) -> None:
            super().__init__()
            self.norm1 = RMSNorm(config.hidden_size)
            self.norm2 = RMSNorm(config.hidden_size)
            self.attn = SelfAttention(config)
            self.ffn = MoEFeedForward(config) if use_moe else DenseFFN(config)
            self.use_moe = use_moe

        def forward(self, x: Tensor) -> tuple[Tensor, list[RouterStats]]:
            x = x + self.attn(self.norm1(x))
            stats: list[RouterStats] = []
            ffn_input = self.norm2(x)
            if self.use_moe:
                ffn_out, router_stats = self.ffn(ffn_input)
                stats.append(router_stats)
            else:
                ffn_out = self.ffn(ffn_input)
            x = x + ffn_out
            return x, stats


    class TinyMoEModel(nn.Module):
        def __init__(self, config: TinyMoEConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.pos_embed = nn.Embedding(config.context_length, config.hidden_size)
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(config, use_moe=((idx + 1) % config.moe_every_n_layers == 0))
                    for idx in range(config.num_layers)
                ]
            )
            self.norm = RMSNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embed.weight

        def forward(
            self,
            input_ids: Tensor,
            labels: Tensor | None = None,
        ) -> dict[str, Any]:
            bsz, seq_len = input_ids.shape
            pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
            pos = pos.clamp_max(self.config.context_length - 1)
            x = self.token_embed(input_ids) + self.pos_embed(pos)
            aux_loss = torch.tensor(0.0, device=input_ids.device)
            z_loss = torch.tensor(0.0, device=input_ids.device)
            dropped_tokens = 0
            expert_load_accum: list[list[float]] = []

            for layer in self.layers:
                x, stats_list = layer(x)
                for stats in stats_list:
                    aux_loss = aux_loss + torch.tensor(stats.aux_loss, device=input_ids.device)
                    z_loss = z_loss + torch.tensor(stats.z_loss, device=input_ids.device)
                    dropped_tokens += stats.dropped_tokens
                    expert_load_accum.append(stats.expert_load)

            x = self.norm(x)
            logits = self.lm_head(x)
            output: dict[str, Any] = {
                "logits": logits,
                "aux_loss": aux_loss,
                "z_loss": z_loss,
                "dropped_tokens": dropped_tokens,
                "expert_load": _average_expert_load(expert_load_accum),
            }

            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                output["loss"] = loss + 0.01 * aux_loss + 0.001 * z_loss

            return output


else:

    class TinyMoEModel:  # type: ignore[override]
        def __init__(self, config: TinyMoEConfig) -> None:
            raise RuntimeError("PyTorch is required for TinyMoEModel. Install with: pip install .[train]")


def _average_expert_load(values: list[list[float]]) -> list[float]:
    if not values:
        return []
    experts = len(values[0])
    out = [0.0 for _ in range(experts)]
    for row in values:
        for idx, value in enumerate(row):
            out[idx] += value
    return [value / len(values) for value in out]
