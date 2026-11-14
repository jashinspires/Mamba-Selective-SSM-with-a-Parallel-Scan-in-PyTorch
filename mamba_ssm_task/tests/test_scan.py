"""Tests for Phase 2 & 3: Sequential recurrence, associative scan, and official-import ban."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.mamba.discretize import discretize
from src.mamba.scan import (
    associative_scan,
    compose,
    measure_scan_equivalence,
    parallel_scan,
    sequential_scan,
)


def test_sequential_one_state() -> None:
    A = torch.tensor([[-1.0]], dtype=torch.float32)
    delta = torch.tensor([[[[math.log(2.0)]], [[math.log(2.0)]]]], dtype=torch.float32)
    B = torch.tensor([[[[2.0]], [[2.0]]]], dtype=torch.float32)
    A_bar, B_bar = discretize(delta, A, B)
    x = torch.tensor([[[3.0], [1.0]]], dtype=torch.float32)
    C = torch.ones_like(A_bar)

    y_seq = sequential_scan(A_bar, B_bar, x, C)
    y_par = parallel_scan(A_bar, B_bar, x, C)
    expected = torch.tensor([[[3.0], [2.5]]], dtype=torch.float32)
    assert torch.allclose(y_seq, expected, atol=1e-5)
    assert torch.allclose(y_par, expected, atol=1e-5)


def test_compose_associative() -> None:
    gen = torch.Generator().manual_seed(1234)
    p1 = (torch.rand((4, 8), generator=gen), torch.randn((4, 8), generator=gen))
    p2 = (torch.rand((4, 8), generator=gen), torch.randn((4, 8), generator=gen))
    p3 = (torch.rand((4, 8), generator=gen), torch.randn((4, 8), generator=gen))

    left = compose(p3, compose(p2, p1))
    right = compose(compose(p3, p2), p1)
    assert torch.allclose(left[0], right[0], atol=1e-5)
    assert torch.allclose(left[1], right[1], atol=1e-5)


def test_four_step() -> None:
    a = torch.tensor([0.5, 0.5, 0.25, 0.5], dtype=torch.float32).view(1, 4, 1, 1)
    b = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32).view(1, 4, 1, 1)
    x = torch.ones((1, 4, 1), dtype=torch.float32)
    C = torch.ones_like(a)

    expected = torch.tensor([1.0, 1.5, 1.375, 1.6875], dtype=torch.float32).view(1, 4, 1)
    y_seq = sequential_scan(a, b, x, C)
    _, b_out = associative_scan(a, b)
    y_par = parallel_scan(a, b, x, C)

    assert torch.allclose(y_seq, expected, atol=1e-5)
    assert torch.allclose(b_out.view(1, 4, 1), expected, atol=1e-5)
    assert torch.allclose(y_par, expected, atol=1e-5)


TASK_ROOT = Path(__file__).resolve().parent.parent


def test_scan_equivalence_and_report() -> None:
    report_path = TASK_ROOT / "reports" / "scan_equivalence.json"
    report = measure_scan_equivalence(output_path=report_path)
    assert report["length_8_max_abs_err"] <= 1e-5
    assert report["length_128_max_abs_err"] <= 1e-3

    saved = json.loads(report_path.read_text())
    assert saved["length_8_max_abs_err"] <= 1e-5
    assert saved["length_128_max_abs_err"] <= 1e-3


def test_no_official_import() -> None:
    import ast

    forbidden_modules = ("mamba_ssm", "selective_scan_fn", "selective_scan_cuda", "reference")
    forbidden_text = ("mamba_ssm", "selective_scan_fn", "selective_scan_cuda")
    src_files = list((TASK_ROOT / "src").rglob("*.py"))
    assert len(src_files) > 0
    for py_file in src_files:
        text = py_file.read_text()
        for token in forbidden_text:
            assert token not in text, f"Forbidden token {token!r} found in {py_file}"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for mod in forbidden_modules:
                        assert not alias.name.startswith(mod), f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                for mod in forbidden_modules:
                    assert not node.module.startswith(mod), f"Forbidden import from {node.module} in {py_file}"


