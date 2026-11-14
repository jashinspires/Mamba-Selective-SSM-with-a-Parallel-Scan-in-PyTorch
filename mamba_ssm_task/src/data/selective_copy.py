"""Selective copying synthetic memory benchmark generator."""

from __future__ import annotations

import torch

NOISE_TOKEN = 0
NUM_CONTENT_TOKENS = 16
RECALL_MARKER = 17
SELECTIVE_COPY_VOCAB_SIZE = 18


def make_selective_copy(
    batch: int,
    length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic selective-copying sequences.

    Vocabulary:
        0: noise
        1..16: content tokens
        17: recall marker

    Args:
        batch: Number of sequences in the batch.
        length: Sequence length L (must be >= 32 for 16 prefix content tokens + 16 recall steps).
        seed: Deterministic RNG seed.

    Returns:
        Tuple (input_ids, target_ids, loss_mask) of shapes (batch, length).
        loss_mask is 1.0 only on the last 16 recall positions.
    """
    if length < 32:
        raise ValueError(f"Selective copy requires length >= 32, got {length}.")

    n_content = 16
    prefix_len = length - n_content
    gen = torch.Generator().manual_seed(seed)

    input_ids = torch.zeros((batch, length), dtype=torch.long)
    target_ids = torch.zeros((batch, length), dtype=torch.long)
    loss_mask = torch.zeros((batch, length), dtype=torch.float32)

    for b in range(batch):
        # Sample 16 distinct content positions without replacement in the prefix.
        positions = torch.randperm(prefix_len, generator=gen)[:n_content].sort().values
        tokens = torch.randint(1, NUM_CONTENT_TOKENS + 1, (n_content,), generator=gen, dtype=torch.long)
        input_ids[b, positions] = tokens
        input_ids[b, prefix_len:] = RECALL_MARKER
        target_ids[b, prefix_len:] = tokens
        loss_mask[b, prefix_len:] = 1.0

    return input_ids, target_ids, loss_mask
