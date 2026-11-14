"""Tests for Phase 4: MambaBlock structure, causality, gate behavior, and step equivalence."""

from __future__ import annotations

import torch

from src.mamba.block import MambaBlock


def test_block_defaults_and_shapes() -> None:
    block = MambaBlock(d_model=64)
    assert block.d_state == 16
    assert block.expand == 2
    assert block.conv_kernel == 4
    assert block.d_inner == 128

    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)


def test_block_causality() -> None:
    torch.manual_seed(7)
    block = MambaBlock(d_model=32, d_state=8, expand=2, conv_kernel=4)
    block.eval()

    x1 = torch.randn(2, 20, 32)
    x2 = x1.clone()
    t_split = 10
    x2[:, t_split + 1 :, :] = torch.randn(2, 20 - (t_split + 1), 32) * 5.0

    with torch.no_grad():
        y1 = block(x1)
        y2 = block(x2)

    assert torch.allclose(y1[:, : t_split + 1, :], y2[:, : t_split + 1, :], atol=1e-6)


def test_zeroing_gate_zeros_output() -> None:
    torch.manual_seed(11)
    block = MambaBlock(d_model=32, d_state=8, expand=2, conv_kernel=4)
    with torch.no_grad():
        # Second half of in_proj weights produces the gate branch.
        block.in_proj.weight[block.d_inner :, :].zero_()

    x = torch.randn(2, 12, 32)
    out = block(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_block_step_matches_forward() -> None:
    torch.manual_seed(29)
    block = MambaBlock(d_model=32, d_state=8, expand=2, conv_kernel=4)
    block.eval()

    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_full = block(x)
        ssm_state, conv_state = block.init_state(batch_size=2)
        step_outs = []
        for t in range(x.shape[1]):
            y_t, ssm_state, conv_state = block.step(x[:, t, :], ssm_state, conv_state)
            step_outs.append(y_t)
        y_step = torch.stack(step_outs, dim=1)

    assert torch.allclose(y_full, y_step, atol=1e-5)
