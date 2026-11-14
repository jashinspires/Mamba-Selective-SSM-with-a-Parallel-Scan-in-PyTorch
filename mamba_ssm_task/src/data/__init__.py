"""Deterministic synthetic sequence-memory task generators."""

from src.data.induction import make_induction
from src.data.selective_copy import make_selective_copy

__all__ = ["make_induction", "make_selective_copy"]
