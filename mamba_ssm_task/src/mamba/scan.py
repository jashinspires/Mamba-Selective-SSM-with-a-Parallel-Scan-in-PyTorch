"""Sequential reference recurrence and parallel associative scan in pure PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def sequential_scan(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    x: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Reference sequential recurrence starting from zero hidden state.

    Args:
        A_bar: Discrete transition factors, shape (batch, length, d_inner, d_state).
        B_bar: Discrete input factors, shape (batch, length, d_inner, d_state).
        x: Input sequence, shape (batch, length, d_inner).
        C: Output projection factors, shape (batch, length, d_inner, d_state).

    Returns:
        Output sequence y of shape (batch, length, d_inner).
    """
    batch, length, d_inner, d_state = A_bar.shape
    # Initial hidden state before the first token is zero.
    h = torch.zeros((batch, d_inner, d_state), device=x.device, dtype=x.dtype)
    outputs: list[torch.Tensor] = []

    for t in range(length):
        # Broadcast x[:, t] across the state dimension inside the per-step update.
        x_t = x[:, t].unsqueeze(-1)
        h = A_bar[:, t] * h + B_bar[:, t] * x_t
        y_t = (C[:, t] * h).sum(dim=-1)
        outputs.append(y_t)

    if not outputs:
        return torch.zeros((batch, 0, d_inner), device=x.device, dtype=x.dtype)
    return torch.stack(outputs, dim=1)


def compose(
    later: tuple[torch.Tensor, torch.Tensor],
    earlier: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative span composition operator for linear recurrences.

    Composing a later span (a2, b2) after an earlier span (a1, b1) yields
    (a2 * a1, a2 * b1 + b2).
    """
    later_a, later_b = later
    earlier_a, earlier_b = earlier
    combined_a = later_a * earlier_a
    combined_b = later_a * earlier_b + later_b
    return combined_a, combined_b


def associative_scan(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Work-efficient Blelloch inclusive parallel prefix scan along dim=1.

    Args:
        a: Multiplicative coefficients, shape (batch, length, ...).
        b: Additive state contributions, shape (batch, length, ...).

    Returns:
        Inclusive prefix pairs (a_out, b_out) where b_out equals the hidden state h[t].
    """
    batch, length = a.shape[0], a.shape[1]
    if length <= 1:
        return a, b

    pairs = length // 2
    even_a = a[:, 0 : 2 * pairs : 2]
    even_b = b[:, 0 : 2 * pairs : 2]
    odd_a = a[:, 1 : 2 * pairs : 2]
    odd_b = b[:, 1 : 2 * pairs : 2]

    # Up-sweep: compose each odd element after its preceding even element.
    reduced_a, reduced_b = compose((odd_a, odd_b), (even_a, even_b))
    scan_odd_a, scan_odd_b = associative_scan(reduced_a, reduced_b)

    # Down-sweep: even positions t >= 2 compose original even[t] after scan_odd[t//2 - 1].
    if length % 2 == 0:
        prev_odd_a = scan_odd_a[:, :-1]
        prev_odd_b = scan_odd_b[:, :-1]
    else:
        prev_odd_a = scan_odd_a
        prev_odd_b = scan_odd_b

    tail_even_a, tail_even_b = compose((a[:, 2::2], b[:, 2::2]), (prev_odd_a, prev_odd_b))
    scan_even_a = torch.cat([a[:, :1], tail_even_a], dim=1)
    scan_even_b = torch.cat([b[:, :1], tail_even_b], dim=1)

    inter_a = torch.stack([scan_even_a[:, :pairs], scan_odd_a], dim=2).reshape(
        batch, 2 * pairs, *a.shape[2:]
    )
    inter_b = torch.stack([scan_even_b[:, :pairs], scan_odd_b], dim=2).reshape(
        batch, 2 * pairs, *b.shape[2:]
    )

    if length % 2 == 1:
        inter_a = torch.cat([inter_a, scan_even_a[:, -1:]], dim=1)
        inter_b = torch.cat([inter_b, scan_even_b[:, -1:]], dim=1)

    return inter_a, inter_b



def _chunked_scan_fwd(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
    """Compute inclusive linear recurrence h[t] = a[t]*h[t-1] + b[t] using chunked associative scan."""
    batch, length = a.shape[0], a.shape[1]
    if length < chunk_size * 2:
        _, h = associative_scan(a, b)
        return h

    pad = (chunk_size - length % chunk_size) % chunk_size
    if pad:
        a_pad = torch.cat([a, torch.ones((batch, pad, *a.shape[2:]), device=a.device, dtype=a.dtype)], dim=1)
        b_pad = torch.cat([b, torch.zeros((batch, pad, *b.shape[2:]), device=b.device, dtype=b.dtype)], dim=1)
    else:
        a_pad, b_pad = a, b

    n_chunks = a_pad.shape[1] // chunk_size
    ac = a_pad.view(batch, n_chunks, chunk_size, *a.shape[2:])
    bc = b_pad.view(batch, n_chunks, chunk_size, *b.shape[2:])

    la_list = [ac[:, :, 0]]
    lb_list = [bc[:, :, 0]]
    for i in range(1, chunk_size):
        na, nb = compose((ac[:, :, i], bc[:, :, i]), (la_list[-1], lb_list[-1]))
        la_list.append(na)
        lb_list.append(nb)
    la = torch.stack(la_list, dim=2)
    lb = torch.stack(lb_list, dim=2)

    _, carry_b = associative_scan(la[:, :, -1], lb[:, :, -1])
    prev_carry_b = torch.cat([torch.zeros_like(carry_b[:, :1]), carry_b[:, :-1]], dim=1).unsqueeze(2)
    out_b = (la * prev_carry_b + lb).view(batch, n_chunks * chunk_size, *b.shape[2:])
    return out_b[:, :length]


class _AssociativeScanFn(torch.autograd.Function):
    """Memory-efficient autograd wrapper using chunked associative_scan for forward and reverse adjoint."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = _chunked_scan_fwd(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, h = ctx.saved_tensors
        # Reverse linear recurrence: dh[t] = a[t+1] * dh[t+1] + grad_h[t]
        a_next = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], dim=1)
        dh = _chunked_scan_fwd(a_next.flip(1), grad_h.flip(1)).flip(1)
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return dh * h_prev, dh



def parallel_scan(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    x: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Compute the selective SSM recurrence via parallel associative scan.

    Args:
        A_bar: Discrete transition factors, shape (batch, length, d_inner, d_state).
        B_bar: Discrete input factors, shape (batch, length, d_inner, d_state).
        x: Input sequence, shape (batch, length, d_inner).
        C: Output projection factors, shape (batch, length, d_inner, d_state).

    Returns:
        Output sequence y of shape (batch, length, d_inner).
    """
    if x.shape[1] == 0:
        return torch.zeros_like(x)
    # Form per-timestep additive inputs b[t] = B_bar[t] * x[t].
    b = B_bar * x.unsqueeze(-1)
    h = _AssociativeScanFn.apply(A_bar, b)
    y = (C * h).sum(dim=-1)
    return y



def measure_scan_equivalence(
    output_path: str | Path = "reports/scan_equivalence.json",
    seed: int = 20261114,
) -> dict[str, float]:
    """Verify sequential vs parallel scan numerical agreement at lengths 8 and 128."""
    gen = torch.Generator().manual_seed(seed)

    def _max_err(batch: int, length: int, d_inner: int, d_state: int) -> float:
        a_bar = torch.rand((batch, length, d_inner, d_state), generator=gen, dtype=torch.float32) * 0.9 + 0.05
        b_bar = torch.randn((batch, length, d_inner, d_state), generator=gen, dtype=torch.float32) * 0.2
        x = torch.randn((batch, length, d_inner), generator=gen, dtype=torch.float32)
        c = torch.randn((batch, length, d_inner, d_state), generator=gen, dtype=torch.float32) * 0.2
        y_seq = sequential_scan(a_bar, b_bar, x, c)
        y_par = parallel_scan(a_bar, b_bar, x, c)
        return float((y_seq - y_par).abs().max().item())

    err_len_8 = _max_err(batch=2, length=8, d_inner=16, d_state=8)
    err_len_128 = _max_err(batch=2, length=128, d_inner=32, d_state=16)

    report = {
        "length_8_max_abs_err": err_len_8,
        "length_128_max_abs_err": err_len_128,
        "length_8_tolerance": 1e-5,
        "length_128_tolerance": 1e-3,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report
