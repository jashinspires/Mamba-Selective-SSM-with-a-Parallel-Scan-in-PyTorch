"""Causal Transformer baseline matched in depth, width, and parameter count to MambaLM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerKVCache:
    """Explicit key-value cache storing per-layer K and V tensors across generated tokens."""

    layers: list[tuple[torch.Tensor, torch.Tensor]]

    def byte_size(self) -> int:
        """Return total bytes occupied by cached key and value tensors."""
        total = 0
        for k_cache, v_cache in self.layers:
            total += k_cache.numel() * k_cache.element_size()
            total += v_cache.numel() * v_cache.element_size()
        return total


def sinusoidal_positions(seq_len: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute non-parametric sinusoidal positional encodings for arbitrary length."""
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros((seq_len, d_model), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.to(dtype=dtype)


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention with explicit KV cache support."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.out_proj(attn_out)

    def step(
        self,
        x_step: torch.Tensor,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x_step.shape[0]
        q = self.q_proj(x_step).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_step).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_step).view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)

        if k_cache is None or v_cache is None:
            k_all = k_new
            v_all = v_new
        else:
            k_all = torch.cat([k_cache, k_new], dim=2)
            v_all = torch.cat([v_cache, v_new], dim=2)

        attn_out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, self.d_model)
        return self.out_proj(attn_out), k_all, v_all


class TransformerBlock(nn.Module):
    """Pre-LN causal transformer block with standard MLP."""

    def __init__(self, d_model: int, n_heads: int, mlp_width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(
        self,
        x_step: torch.Tensor,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, next_k, next_v = self.attn.step(self.norm1(x_step), k_cache, v_cache)
        x_step = x_step + attn_out
        x_step = x_step + self.mlp(self.norm2(x_step))
        return x_step, next_k, next_v


class CausalTransformerLM(nn.Module):
    """Two-layer causal Transformer baseline with untied embedding and linear head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_width: int = 240,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_width = mlp_width

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, mlp_width=mlp_width) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, length = input_ids.shape
        x = self.embedding(input_ids)
        pe = sinusoidal_positions(length, self.d_model, x.device, x.dtype)
        x = x + pe.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm_f(x))

    def init_kv_cache(self) -> TransformerKVCache:
        """Create an empty KV cache before stepping through tokens."""
        return TransformerKVCache(layers=[])

    def step(
        self,
        kv_cache: TransformerKVCache,
        token: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, TransformerKVCache]:
        """Step causal transformer by one token and append K, V to the growing cache."""
        token_ids = token.squeeze(-1) if token.dim() == 2 else token
        x = self.embedding(token_ids)
        pe = sinusoidal_positions(pos + 1, self.d_model, x.device, x.dtype)[pos]
        x = x + pe.unsqueeze(0)

        new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            k_prev, v_prev = kv_cache.layers[idx] if idx < len(kv_cache.layers) else (None, None)
            x, k_next, v_next = layer.step(x, k_prev, v_prev)
            new_layers.append((k_next, v_next))

        logits = self.lm_head(self.norm_f(x))
        return logits, TransformerKVCache(layers=new_layers)


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters (frozen tensors excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_matched_mlp_width(
    mamba_param_count: int,
    vocab_size: int,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
) -> int:
    """Select Transformer MLP width so trainable parameter count matches Mamba within 10%."""
    best_width = 4 * d_model
    best_diff = float("inf")
    for width in range(64, 513, 4):
        candidate = CausalTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_width=width,
        )
        tf_count = count_trainable_params(candidate)
        diff = abs(tf_count - mamba_param_count)
        if diff < best_diff:
            best_diff = diff
            best_width = width
    return best_width
