"""Tests for Phase 5 & 11: Selective copy and induction generators and loss masks."""

from __future__ import annotations

import torch

from src.data.induction import INDUCTION_VOCAB_SIZE, make_induction
from src.data.selective_copy import (
    NOISE_TOKEN,
    RECALL_MARKER,
    make_selective_copy,
)
from src.train import masked_cross_entropy, masked_token_accuracy


def test_selective_copy_generator() -> None:
    x1, y1, m1 = make_selective_copy(batch=4, length=256, seed=42)
    x2, y2, m2 = make_selective_copy(batch=4, length=256, seed=42)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(m1, m2)

    assert x1.shape == (4, 256)
    assert y1.shape == (4, 256)
    assert m1.shape == (4, 256)

    for b in range(4):
        prefix = x1[b, :-16]
        suffix = x1[b, -16:]
        assert torch.all(suffix == RECALL_MARKER)
        content = prefix[prefix != NOISE_TOKEN]
        assert len(content) == 16
        assert torch.all((content >= 1) & (content <= 16))
        assert torch.equal(y1[b, -16:], content)
        assert torch.all(m1[b, :-16] == 0)
        assert torch.all(m1[b, -16:] == 1)


def test_induction_generator() -> None:
    x, y, m = make_induction(batch=8, length=256, seed=99)
    assert x.shape == (8, 256)
    assert y.shape == (8, 256)
    assert m.shape == (8, 256)
    assert torch.all((x >= 0) & (x < INDUCTION_VOCAB_SIZE))

    for b in range(8):
        assert torch.all(m[b, :-1] == 0)
        assert m[b, -1] == 1
        query_key = int(x[b, -1].item())
        expected_val = int(y[b, -1].item())
        first_half = x[b, :128].tolist()
        matches = [first_half[i + 1] for i in range(len(first_half) - 1) if first_half[i] == query_key]
        assert expected_val in matches


def test_loss_mask() -> None:
    # Verify unmasked positions do not affect masked loss or masked accuracy.
    logits = torch.randn(2, 10, 18)
    targets = torch.randint(0, 18, (2, 10))
    mask = torch.zeros(2, 10)
    mask[:, -2:] = 1.0

    loss_a = masked_cross_entropy(logits, targets, mask)
    acc_a = masked_token_accuracy(logits, targets, mask)

    logits_perturbed = logits.clone()
    logits_perturbed[:, :-2, :] += 100.0
    targets_perturbed = targets.clone()
    targets_perturbed[:, :-2] = (targets_perturbed[:, :-2] + 1) % 18

    loss_b = masked_cross_entropy(logits_perturbed, targets_perturbed, mask)
    acc_b = masked_token_accuracy(logits_perturbed, targets_perturbed, mask)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    assert acc_a == acc_b
