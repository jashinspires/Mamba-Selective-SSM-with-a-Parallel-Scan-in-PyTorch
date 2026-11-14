"""Induction head synthetic associative-recall benchmark generator."""

from __future__ import annotations

import torch

INDUCTION_VOCAB_SIZE = 16
NUM_BIGRAMS = 4


def make_induction(
    batch: int,
    length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic induction-head associative recall sequences.

    Vocabulary size is 16 (ids 0..15). Four distinct key-value bigrams are placed
    in the first half of each sequence; other early positions are filled with
    non-key tokens. The final token is one of the four keys, and the target at
    the final position is the value that followed that key in its bigram.

    Args:
        batch: Number of sequences in the batch.
        length: Sequence length L (must be >= 16).
        seed: Deterministic RNG seed.

    Returns:
        Tuple (input_ids, target_ids, loss_mask) of shapes (batch, length).
        loss_mask is 1.0 only at the last index.
    """
    if length < 16:
        raise ValueError(f"Induction requires length >= 16, got {length}.")

    half_len = length // 2
    # Sample bigram start positions from even indices in the first half so bigrams never overlap.
    candidate_starts = torch.arange(0, max(8, half_len - 1), 2)
    gen = torch.Generator().manual_seed(seed)

    input_ids = torch.zeros((batch, length), dtype=torch.long)
    target_ids = torch.zeros((batch, length), dtype=torch.long)
    loss_mask = torch.zeros((batch, length), dtype=torch.float32)

    for b in range(batch):
        vocab_perm = torch.randperm(INDUCTION_VOCAB_SIZE, generator=gen)
        keys = vocab_perm[:NUM_BIGRAMS]
        non_keys = vocab_perm[NUM_BIGRAMS:]

        # Draw values from non-key tokens so keys never occur ambiguously inside filler or values.
        val_indices = torch.randint(0, len(non_keys), (NUM_BIGRAMS,), generator=gen)
        values = non_keys[val_indices]

        # Fill all early positions (0 .. length - 2) with non-key tokens.
        filler_idx = torch.randint(0, len(non_keys), (length - 1,), generator=gen)
        input_ids[b, : length - 1] = non_keys[filler_idx]

        # Place the 4 distinct key-value bigrams in the first half.
        pos_perm = torch.randperm(len(candidate_starts), generator=gen)[:NUM_BIGRAMS]
        starts = candidate_starts[pos_perm]
        for idx in range(NUM_BIGRAMS):
            p = int(starts[idx].item())
            input_ids[b, p] = keys[idx]
            input_ids[b, p + 1] = values[idx]

        # Place one of the four keys at the final position and target its paired value.
        query_idx = int(torch.randint(0, NUM_BIGRAMS, (1,), generator=gen).item())
        input_ids[b, length - 1] = keys[query_idx]
        target_ids[b, length - 1] = values[query_idx]
        loss_mask[b, length - 1] = 1.0

    return input_ids, target_ids, loss_mask
