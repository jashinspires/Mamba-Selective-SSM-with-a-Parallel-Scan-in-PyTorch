"""Tests for Phase 1: ZOH discretization and selective Delta positivity."""

from __future__ import annotations

import math

import torch

from src.mamba.block import MambaBlock
from src.mamba.discretize import discretize


def test_one_state_discretize() -> None:
    A = torch.tensor([[-1.0]], dtype=torch.float32)
    delta = torch.tensor([[[[math.log(2.0)]]]], dtype=torch.float32)
    B = torch.tensor([[[[2.0]]]], dtype=torch.float32)

    A_bar, B_bar = discretize(delta, A, B)
    assert torch.allclose(A_bar, torch.tensor([[[[0.5]]]]), atol=1e-5)
    assert torch.allclose(B_bar, torch.tensor([[[[1.0]]]]), atol=1e-5)


def test_small_delta_limit() -> None:
    A = torch.tensor([[-1.0]], dtype=torch.float32)
    delta = torch.tensor([[[[1e-3]]]], dtype=torch.float32)
    B = torch.tensor([[[[3.5]]]], dtype=torch.float32)

    A_bar, B_bar = discretize(delta, A, B)
    assert torch.allclose(A_bar, 1.0 + delta * A, atol=1e-5)
    assert torch.allclose(B_bar, delta * B, atol=1e-4)


def test_contractive_a_bar_and_positive_delta() -> None:
    torch.manual_seed(0)
    block = MambaBlock(d_model=16, d_state=8, expand=2, conv_kernel=4)
    x = torch.randn(2, 12, 16)
    delta = torch.nn.functional.softplus(block.dt_proj(torch.randn(2, 12, block.d_inner))).unsqueeze(-1)
    assert torch.all(delta > 0.0)

    A = -torch.exp(block.A_log)
    assert torch.all(A < 0.0)
    B = torch.randn(2, 12, 1, 8)
    A_bar, _ = discretize(delta, A, B)
    assert torch.all((A_bar > 0.0) & (A_bar < 1.0))
