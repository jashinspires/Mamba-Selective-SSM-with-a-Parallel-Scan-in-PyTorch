"""Two-layer Mamba language model and constant-memory recurrent stepping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.mamba.block import MambaBlock


@dataclass
class MambaState:
    """Recurrent inference state containing SSM states and fixed conv tails per layer."""

    layer_states: list[tuple[torch.Tensor, torch.Tensor]]
    model: "MambaLM | None" = None

    def byte_size(self) -> int:
        """Return total bytes occupied by SSM hidden states and convolution tails."""
        total = 0
        for ssm_state, conv_state in self.layer_states:
            total += ssm_state.numel() * ssm_state.element_size()
            total += conv_state.numel() * conv_state.element_size()
        return total


class ResidualMambaLayer(nn.Module):
    """Pre-norm residual wrapper around a MambaBlock."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_kernel: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = MambaBlock(d_model=d_model, d_state=d_state, expand=expand, conv_kernel=conv_kernel)

    def forward(self, x: torch.Tensor, use_parallel_scan: bool = True) -> torch.Tensor:
        return x + self.block(self.norm(x), use_parallel_scan=use_parallel_scan)

    def step(
        self,
        x_step: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x_step
        out, new_ssm, new_conv = self.block.step(self.norm(x_step), ssm_state, conv_state)
        return residual + out, new_ssm, new_conv


class MambaLM(nn.Module):
    """Token embedding + stack of Mamba blocks + LayerNorm + linear logit head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.expand = expand
        self.conv_kernel = conv_kernel

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                ResidualMambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    conv_kernel=conv_kernel,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, use_parallel_scan: bool = True) -> torch.Tensor:
        """Compute vocabulary logits of shape (batch, length, vocab_size)."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, use_parallel_scan=use_parallel_scan)
        x = self.norm_f(x)
        return self.lm_head(x)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MambaState:
        """Initialize zero recurrent state (SSM hidden state + 3-token conv tail) for all layers."""
        layer_states = [
            layer.block.init_state(batch_size=batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]
        return MambaState(layer_states=layer_states, model=self)

    def step(
        self,
        state: MambaState,
        token: torch.Tensor,
    ) -> tuple[torch.Tensor, MambaState]:
        """Step the full MambaLM forward by one token using constant recurrent state."""
        if token.dim() == 2:
            token_ids = token.squeeze(-1)
        else:
            token_ids = token

        x = self.embedding(token_ids)
        new_layer_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (ssm_state, conv_state) in zip(self.layers, state.layer_states):
            x, new_ssm, new_conv = layer.step(x, ssm_state, conv_state)
            new_layer_states.append((new_ssm, new_conv))

        logits = self.lm_head(self.norm_f(x))
        return logits, MambaState(layer_states=new_layer_states, model=state.model or self)


def step_mamba(
    state: MambaState,
    token: torch.Tensor,
    model: MambaLM | None = None,
) -> tuple[torch.Tensor, MambaState]:
    """Advance Mamba inference by one token without storing hidden-state history."""
    target_model = model or state.model
    if target_model is None:
        raise ValueError("step_mamba requires a MambaState attached to a model or an explicit model argument.")
    return target_model.step(state, token)
