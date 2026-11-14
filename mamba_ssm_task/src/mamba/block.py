"""Mamba selective state-space block with causal depthwise conv1d and SiLU gate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mamba.discretize import discretize
from src.mamba.scan import parallel_scan, sequential_scan


class MambaBlock(nn.Module):
    """Selective SSM block: input projection, causal depthwise conv1d, SiLU, SSM, gate, output proj."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.d_inner = expand * d_model

        # Lift d_model to 2 * d_inner for sequence branch and independent gate branch.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise causal conv1d: pad kernel - 1 on the left and crop output to input length.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
            bias=True,
        )

        # Learned log-parameterized negative diagonal A (S4D-Real initialization).
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))

        # Input-dependent projections for Delta (with bias), B, and C.
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Initialize dt bias so initial softplus(dt) spans ~[0.001, 0.1] for varied memory horizons.
        with torch.no_grad():
            dt_init = torch.exp(
                torch.empty(self.d_inner).uniform_(math.log(0.001), math.log(0.1))
            )
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_softplus)

        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # Output projection without bias so zeroing the gate branch zeros the block output.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, use_parallel_scan: bool = True) -> torch.Tensor:
        """Run full-sequence Mamba block.

        Args:
            x: Tensor of shape (batch, length, d_model).
            use_parallel_scan: Whether to use parallel associative scan instead of sequential loop.

        Returns:
            Tensor of shape (batch, length, d_model).
        """
        _, length, _ = x.shape
        projected = self.in_proj(x)
        x_seq, gate_branch = projected.chunk(2, dim=-1)

        # Depthwise causal convolution over the sequence branch only.
        x_conv = self.conv1d(x_seq.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_act = F.silu(x_conv)

        # Input-dependent step size and input/output state couplings.
        delta = F.softplus(self.dt_proj(x_act)).unsqueeze(-1)
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_act).unsqueeze(2)
        C = self.C_proj(x_act).unsqueeze(2).expand(-1, -1, self.d_inner, -1)

        A_bar, B_bar = discretize(delta, A, B)
        scan_fn = parallel_scan if use_parallel_scan else sequential_scan
        y_ssm = scan_fn(A_bar, B_bar, x_act, C)

        # Gate multiplies the SSM output after the recurrence.
        gated = y_ssm * F.silu(gate_branch)
        return self.out_proj(gated)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate initial zero SSM hidden state and causal convolution tail."""
        ssm_state = torch.zeros((batch_size, self.d_inner, self.d_state), device=device, dtype=dtype)
        conv_state = torch.zeros((batch_size, self.d_inner, self.conv_kernel - 1), device=device, dtype=dtype)
        return ssm_state, conv_state

    def step(
        self,
        x_step: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the Mamba block by a single token in constant state memory.

        Args:
            x_step: Input activation for one token, shape (batch, d_model).
            ssm_state: Current hidden state, shape (batch, d_inner, d_state).
            conv_state: Last (conv_kernel - 1) sequence-branch vectors, shape (batch, d_inner, 3).

        Returns:
            Tuple (y_out, new_ssm_state, new_conv_state).
        """
        projected = self.in_proj(x_step)
        x_seq, gate_branch = projected.chunk(2, dim=-1)

        # Shift fixed-length convolution tail and apply depthwise kernel across the window.
        conv_window = torch.cat([conv_state, x_seq.unsqueeze(-1)], dim=-1)
        new_conv_state = conv_window[:, :, 1:].contiguous()
        conv_weight = self.conv1d.weight.squeeze(1)
        conv_out = (conv_window * conv_weight).sum(dim=-1)
        if self.conv1d.bias is not None:
            conv_out = conv_out + self.conv1d.bias
        x_act = F.silu(conv_out)

        # Discretize at this single timestep: delta (B, D, 1), A (D, N), B (B, 1, N).
        delta = F.softplus(self.dt_proj(x_act)).unsqueeze(-1)
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_act).unsqueeze(1)
        C = self.C_proj(x_act).unsqueeze(1)

        A_bar, B_bar = discretize(delta, A, B)
        new_ssm_state = A_bar * ssm_state + B_bar * x_act.unsqueeze(-1)
        y_ssm = (C * new_ssm_state).sum(dim=-1)

        gated = y_ssm * F.silu(gate_branch)
        y_out = self.out_proj(gated)
        return y_out, new_ssm_state, new_conv_state
