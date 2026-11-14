"""Tests for Phase 9 & 12: Constant Mamba state vs growing KV cache and container files."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.mamba.model import MambaLM, step_mamba
from src.transformer_baseline import CausalTransformerLM


TASK_ROOT = Path(__file__).resolve().parent.parent


def test_inference_report_and_plots() -> None:
    report_path = TASK_ROOT / "reports" / "inference.json"
    assert report_path.exists(), "reports/inference.json must exist"
    report = json.loads(report_path.read_text())

    assert report["lengths"] == [256, 1024, 4096]
    mamba_bytes = [int(x) for x in report["mamba_state_bytes"]]
    tf_bytes = [int(x) for x in report["transformer_kv_bytes"]]
    mamba_ms = [float(x) for x in report["mamba_latency_ms"]]
    tf_ms = [float(x) for x in report["transformer_latency_ms"]]

    assert len(mamba_bytes) == 3
    assert len(tf_bytes) == 3
    assert len(mamba_ms) == 3
    assert len(tf_ms) == 3

    assert max(mamba_bytes) / min(mamba_bytes) <= 1.05
    assert tf_bytes[0] < tf_bytes[1] < tf_bytes[2]
    assert tf_bytes[2] / tf_bytes[0] >= 8.0

    mem_plot = TASK_ROOT / "reports" / "inference_memory.png"
    lat_plot = TASK_ROOT / "reports" / "inference_latency.png"
    assert mem_plot.exists() and mem_plot.stat().st_size > 1024
    assert lat_plot.exists() and lat_plot.stat().st_size > 1024


def test_live_step_state_shapes() -> None:
    mamba = MambaLM(vocab_size=18, d_model=64, n_layers=2, d_state=16, expand=2, conv_kernel=4)
    state = mamba.init_state(batch_size=1)
    initial_bytes = state.byte_size()
    tok = torch.tensor([[1]], dtype=torch.long)
    for _ in range(10):
        logits, state = step_mamba(state, tok)
        assert logits.shape == (1, 18)
        for ssm_s, conv_s in state.layer_states:
            assert ssm_s.shape == (1, 128, 16)
            assert conv_s.shape == (1, 128, 3)
    assert state.byte_size() == initial_bytes

    tf = CausalTransformerLM(vocab_size=18, d_model=64, n_layers=2, n_heads=4, mlp_width=240)
    kv = tf.init_kv_cache()
    _, kv_1 = tf.step(kv, tok, pos=0)
    _, kv_2 = tf.step(kv_1, tok, pos=1)
    assert kv_2.byte_size() == 2 * kv_1.byte_size()


def test_container_files() -> None:
    assert (TASK_ROOT / "Dockerfile").exists()
    assert (TASK_ROOT / "docker-compose.yml").exists()
    env_path = TASK_ROOT / ".env.example"
    assert env_path.exists()

    lines = env_path.read_text().splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            assert idx > 0 and lines[idx - 1].strip().startswith("#"), (
                f"Line {idx + 1} ({stripped}) must have a comment on the previous line"
            )

    env_text = env_path.read_text()
    for forbidden in ("BEGIN PRIVATE KEY", "AKIA", "hf_"):
        assert forbidden not in env_text

    readme_text = (TASK_ROOT / "README.md").read_text()
    assert len(readme_text.strip()) > 50
    assert any(tok in readme_text for tok in ("CPU", "GPU", "CUDA", "NVIDIA", "Apple", "x86_64"))

