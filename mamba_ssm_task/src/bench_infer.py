"""Benchmark constant-memory Mamba stepping vs growing Transformer KV cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.selective_copy import SELECTIVE_COPY_VOCAB_SIZE
from src.mamba.model import MambaLM, step_mamba
from src.transformer_baseline import CausalTransformerLM, find_matched_mlp_width, count_trainable_params

BENCH_LENGTHS = [256, 1024, 4096]


def benchmark_mamba_inference(
    model: MambaLM,
    lengths: list[int],
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Step Mamba one token at a time and record state bytes and ms/token."""
    model.eval()
    state_bytes_list: list[int] = []
    latency_ms_list: list[float] = []

    with torch.no_grad():
        # Warmup short sequence.
        warm_state = model.init_state(batch_size=1, device=device)
        warm_tok = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(16):
            _, warm_state = step_mamba(warm_state, warm_tok)

        for length in lengths:
            tokens = torch.randint(0, model.vocab_size, (1, length), dtype=torch.long, device=device)
            state = model.init_state(batch_size=1, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t in range(length):
                _, state = step_mamba(state, tokens[:, t : t + 1])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            state_bytes_list.append(int(state.byte_size()))
            latency_ms_list.append(float(elapsed_ms / length))

    return state_bytes_list, latency_ms_list


def benchmark_transformer_inference(
    model: CausalTransformerLM,
    lengths: list[int],
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Step causal Transformer one token at a time and record KV cache bytes and ms/token."""
    model.eval()
    kv_bytes_list: list[int] = []
    latency_ms_list: list[float] = []

    with torch.no_grad():
        warm_cache = model.init_kv_cache()
        warm_tok = torch.zeros((1, 1), dtype=torch.long, device=device)
        for t in range(16):
            _, warm_cache = model.step(warm_cache, warm_tok, pos=t)

        for length in lengths:
            tokens = torch.randint(0, model.vocab_size, (1, length), dtype=torch.long, device=device)
            kv_cache = model.init_kv_cache()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t in range(length):
                _, kv_cache = model.step(kv_cache, tokens[:, t : t + 1], pos=t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            kv_bytes_list.append(int(kv_cache.byte_size()))
            latency_ms_list.append(float(elapsed_ms / length))

    return kv_bytes_list, latency_ms_list


def save_plots(
    lengths: list[int],
    mamba_bytes: list[int],
    tf_bytes: list[int],
    mamba_ms: list[float],
    tf_ms: list[float],
    out_dir: Path,
) -> None:
    """Render memory and latency comparison PNG plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_labels = [str(l) for l in lengths]
    x_pos = list(range(len(lengths)))

    # 1. Memory bytes plot
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax.plot(x_pos, [b / 1024.0 for b in mamba_bytes], marker="o", linewidth=2.4, color="#2563EB", label="Mamba State + Conv Tail (Constant)")
    ax.plot(x_pos, [b / 1024.0 for b in tf_bytes], marker="s", linewidth=2.4, color="#D97706", label="Transformer KV Cache (Linear Growth)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("State / Cache Memory (KiB)")
    ax.set_title("Recurrent State Memory vs. Growing KV Cache (batch=1, fp32)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "inference_memory.png")
    plt.close(fig)

    # 2. Per-token latency plot
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax.plot(x_pos, mamba_ms, marker="o", linewidth=2.4, color="#2563EB", label="Mamba Recurrent Step")
    ax.plot(x_pos, tf_ms, marker="s", linewidth=2.4, color="#D97706", label="Transformer KV-Cached Step")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Per-Token Latency (ms / token)")
    ax.set_title("Per-Token Autoregressive Stepping Latency")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "inference_latency.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark recurrent Mamba vs Transformer KV cache.")
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "cpu"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "reports"))
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)

    mamba_ckpt = ckpt_dir / "mamba_selective_copy.pt"
    tf_ckpt = ckpt_dir / "transformer_selective_copy.pt"

    if mamba_ckpt.exists() and tf_ckpt.exists():
        mamba_payload = torch.load(mamba_ckpt, map_location=device, weights_only=True)
        tf_payload = torch.load(tf_ckpt, map_location=device, weights_only=True)
        m_cfg = mamba_payload["config"]
        t_cfg = tf_payload["config"]
        mamba = MambaLM(**m_cfg).to(device)
        mamba.load_state_dict(mamba_payload["model_state_dict"])
        transformer = CausalTransformerLM(**t_cfg).to(device)
        transformer.load_state_dict(tf_payload["model_state_dict"])
    else:
        mamba = MambaLM(vocab_size=SELECTIVE_COPY_VOCAB_SIZE, d_model=64, n_layers=2, d_state=16, expand=2, conv_kernel=4).to(device)
        mlp_width = find_matched_mlp_width(count_trainable_params(mamba), SELECTIVE_COPY_VOCAB_SIZE)
        transformer = CausalTransformerLM(vocab_size=SELECTIVE_COPY_VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, mlp_width=mlp_width).to(device)

    mamba_bytes, mamba_ms = benchmark_mamba_inference(mamba, BENCH_LENGTHS, device)
    tf_bytes, tf_ms = benchmark_transformer_inference(transformer, BENCH_LENGTHS, device)

    report = {
        "lengths": BENCH_LENGTHS,
        "device": str(device),
        "mamba_state_bytes": mamba_bytes,
        "transformer_kv_bytes": tf_bytes,
        "mamba_latency_ms": [round(x, 4) for x in mamba_ms],
        "transformer_latency_ms": [round(x, 4) for x in tf_ms],
        "mamba_state_byte_ratio": max(mamba_bytes) / min(mamba_bytes),
        "transformer_4096_over_256_ratio": tf_bytes[-1] / tf_bytes[0],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    save_plots(BENCH_LENGTHS, mamba_bytes, tf_bytes, mamba_ms, tf_ms, out_dir)
    print("Inference benchmark:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
