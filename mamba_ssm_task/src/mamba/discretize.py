"""Zero-order-hold (ZOH) discretization for diagonal continuous SSMs."""

from __future__ import annotations

import torch


def discretize(
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize diagonal negative A and input-dependent B with zero-order hold.

    Args:
        delta: Positive step sizes, shape (batch, length, d_inner, 1) or broadcastable.
        A: Negative diagonal state parameters, shape (d_inner, d_state) or broadcastable.
        B: Input-dependent input matrix, shape (batch, length, d_inner, d_state) or broadcastable.

    Returns:
        Tuple (A_bar, B_bar) where:
            A_bar = exp(delta * A)
            B_bar = ((exp(delta * A) - 1) / A) * B
    """
    # Product delta * A determines exponential decay factor per channel and state dimension.
    delta_a = delta * A
    # Discrete transition multiplier lies in (0, 1) when delta > 0 and A < 0.
    a_bar = torch.exp(delta_a)
    # Use expm1 for accurate small-step behavior: (exp(delta * A) - 1) / A * B.
    b_bar = (torch.expm1(delta_a) / A) * B
    return a_bar, b_bar
