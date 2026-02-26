"""Transformers model implementation for NeuroCoder remote-code loading."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_neurocoder import NeuroCoderConfig
except Exception:
    from configuration_neurocoder import NeuroCoderConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, config: NeuroCoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        x: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape_heads(t: Tensor) -> Tensor:
            return t.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape_heads(q)
        k = shape_heads(k)
        v = shape_heads(v)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None
        key_len = k.shape[-2]
        past_len = key_len - seq_len

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if seq_len > 1 or past_len > 0:
            q_positions = torch.arange(
                past_len,
                past_len + seq_len,
                device=x.device,
            ).unsqueeze(-1)
            k_positions = torch.arange(key_len, device=x.device).unsqueeze(0)
            causal_mask = (k_positions <= q_positions).unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(~causal_mask, float("-inf"))
        if attention_mask is not None:
            # Expect [batch, key_len] style attention mask. Keep only the last key_len
            # columns so generation with cache remains aligned.
            key_mask = attention_mask[:, -key_len:].to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~key_mask, float("-inf"))

        probs = F.softmax(attn, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out(out), present


class DenseFFN(nn.Module):
    def __init__(self, config: NeuroCoderConfig) -> None:
        super().__init__()
        inner = config.hidden_size * config.ffn_multiplier
        self.gate = nn.Linear(config.hidden_size, inner)
        self.up = nn.Linear(config.hidden_size, inner)
        self.down = nn.Linear(inner, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFeedForward(nn.Module):
    def __init__(self, config: NeuroCoderConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.router_top_k
        self.capacity_factor_train = config.capacity_factor_train
        self.capacity_factor_infer = config.capacity_factor_infer
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([DenseFFN(config) for _ in range(config.num_experts)])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        tokens = x_flat.shape[0]

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)

        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_infer
        capacity = max(1, math.ceil(capacity_factor * tokens / self.num_experts))

        output = torch.zeros_like(x_flat)
        expert_load = []

        for expert_id in range(self.num_experts):
            expert = self.experts[expert_id]
            assigned_indices = []
            assigned_weights = []
            for rank in range(self.top_k):
                mask = top_idx[:, rank] == expert_id
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                weights = top_vals[idx, rank]
                assigned_indices.append(idx)
                assigned_weights.append(weights)

            if not assigned_indices:
                expert_load.append(0.0)
                continue

            token_indices = torch.cat(assigned_indices, dim=0)
            token_weights = torch.cat(assigned_weights, dim=0)
            if token_indices.numel() > capacity:
                token_indices = token_indices[:capacity]
                token_weights = token_weights[:capacity]

            expert_in = x_flat[token_indices]
            expert_out = expert(expert_in)
            output[token_indices] += expert_out * token_weights.unsqueeze(-1)
            expert_load.append(float(token_indices.numel() / max(tokens, 1)))

        load_tensor = torch.tensor(expert_load, device=x.device)
        mean_prob = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_prob * load_tensor)
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        return output.reshape(bsz, seq_len, hidden), aux_loss, z_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: NeuroCoderConfig, use_moe: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.attn = SelfAttention(config)
        self.ffn = MoEFeedForward(config) if use_moe else DenseFFN(config)
        self.use_moe = use_moe

    def forward(
        self,
        x: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        attn_out, present = self.attn(
            self.norm1(x),
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        x = x + attn_out
        aux_loss = torch.tensor(0.0, device=x.device)
        z_loss = torch.tensor(0.0, device=x.device)
        ffn_input = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss, z_loss = self.ffn(ffn_input)
        else:
            ffn_out = self.ffn(ffn_input)
        x = x + ffn_out
        return x, aux_loss, z_loss, present


class NeuroCoderForCausalLM(PreTrainedModel):
    config_class = NeuroCoderConfig
    base_model_prefix = "neurocoder"
    _no_split_modules = ["TransformerBlock", "MoEFeedForward"]
    _supports_cache_class = False

    def __init__(self, config: NeuroCoderConfig) -> None:
        super().__init__(config)
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
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        past_key_values = kwargs.get("past_key_values")
        has_past = False
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            has_past = bool(past_key_values.get_seq_length() > 0)
        elif isinstance(past_key_values, tuple) and past_key_values:
            first = past_key_values[0]
            has_past = bool(first and first[0] is not None and first[1] is not None)

        if has_past:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _as_legacy_past_key_values(
        past_key_values: Any,
        num_layers: int,
    ) -> tuple[tuple[Tensor, Tensor] | None, ...]:
        if past_key_values is None:
            return tuple([None] * num_layers)

        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

        if isinstance(past_key_values, list):
            past_key_values = tuple(past_key_values)
        if isinstance(past_key_values, tuple):
            return past_key_values

        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if isinstance(key_cache, list) and isinstance(value_cache, list):
            pairs: list[tuple[Tensor, Tensor] | None] = []
            for idx in range(num_layers):
                if idx < len(key_cache) and idx < len(value_cache):
                    key = key_cache[idx]
                    value = value_cache[idx]
                    if key is not None and value is not None:
                        pairs.append((key, value))
                        continue
                pairs.append(None)
            return tuple(pairs)

        return tuple([None] * num_layers)

    def _reorder_cache(
        self,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | list[tuple[Tensor, Tensor]],
        beam_idx: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        reordered: list[tuple[Tensor, Tensor]] = []
        for key, value in past_key_values:
            reordered.append((key.index_select(0, beam_idx), value.index_select(0, beam_idx)))
        return tuple(reordered)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids is required")

        cache_enabled = bool(self.config.use_cache if use_cache is None else use_cache)
        past = self._as_legacy_past_key_values(past_key_values, len(self.layers))
        bsz, seq_len = input_ids.shape
        past_len = 0
        for entry in past:
            if (
                entry is not None
                and isinstance(entry, tuple)
                and len(entry) == 2
                and entry[0] is not None
                and entry[1] is not None
            ):
                past_len = int(entry[0].shape[2])
                break
        pos = torch.arange(
            past_len,
            past_len + seq_len,
            device=input_ids.device,
        ).unsqueeze(0).expand(bsz, seq_len)
        pos = pos.clamp_max(self.config.context_length - 1)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        aux_loss = torch.tensor(0.0, device=input_ids.device)
        z_loss = torch.tensor(0.0, device=input_ids.device)
        present_key_values: list[tuple[Tensor, Tensor]] = []

        for layer_idx, layer in enumerate(self.layers):
            layer_past = past[layer_idx] if layer_idx < len(past) else None
            x, layer_aux, layer_z, layer_present = layer(
                x,
                past_key_value=layer_past,  # type: ignore[arg-type]
                attention_mask=attention_mask,
                use_cache=cache_enabled,
            )
            aux_loss = aux_loss + layer_aux
            z_loss = z_loss + layer_z
            if cache_enabled and layer_present is not None:
                present_key_values.append(layer_present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + 0.01 * aux_loss + 0.001 * z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(present_key_values) if cache_enabled else None,
        )
