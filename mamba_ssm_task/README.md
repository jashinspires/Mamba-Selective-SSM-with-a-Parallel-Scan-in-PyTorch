# Mamba Selective SSM with a Parallel Scan in PyTorch

Pure PyTorch implementation of Mamba's selective state-space block (`discretize`, `sequential_scan`, associative `compose`, Blelloch/chunked `parallel_scan`, causal depthwise `Conv1d`, and `SiLU` gate), paired with a parameter-matched 2-layer causal Transformer baseline on Selective Copying and Induction Head benchmarks.

## Architecture & Parameter Parity

- **MambaLM**: 2 layers, `d_model=64`, `d_state=16`, `expand=2` (`d_inner=128`), `conv_kernel=4`, untied embedding and linear head (`98,432` trainable parameters on Selective Copying `vocab=18`).
- **CausalTransformerLM**: 2 layers, `d_model=64`, `n_heads=4`, sinusoidal positional encoding, matched `mlp_width=240` (`98,272` trainable parameters on Selective Copying; `0.16%` relative difference, well under the `10%` ceiling).

## Hardware & Environment

- **CPU Unit-Test Verification**: Verified scan equivalence, discretization, block causality, gate zeroing, task generators, and inference memory/latency scaling on CPU (`x86_64`).
- **GPU Training Run**: Update this line with your GPU name (e.g. `NVIDIA T4` / `RTX 3090` / `A100`) and recorded step counts after running `src/train.py` on CUDA.

## Quickstart & Benchmark Commands

```bash
pip install -r requirements.txt

# 1. Run unit tests (no checkpoints required)
pytest tests/test_discretize.py tests/test_scan.py tests/test_block.py tests/test_tasks.py -q

# 2. Train both models on Selective Copying and Induction at length 256 (GPU recommended)
TRAIN_DEVICE=cuda MAX_STEPS=2000 python -m src.train

# 3. Evaluate checkpoints at lengths 256, 1024, and 4096
python -m src.eval

# 4. Benchmark recurrent state memory vs. KV cache and generate PNG plots
python -m src.bench_infer

# 5. Run full checkpoint and inference verification suite
pytest tests/test_checkpoints.py tests/test_infer.py -q
```

## Reports & Interpretation

- `reports/scan_equivalence.json`: Max absolute difference between `sequential_scan` and `parallel_scan` at lengths 8 (`<= 1e-5`) and 128 (`<= 1e-3`).
- `reports/train_eval.json`: Parameter counts, Transformer `mlp_width`, optimizer step counts, and masked token accuracies at lengths `256`, `1024`, and `4096`.
- `reports/inference.json`, `reports/inference_memory.png`, `reports/inference_latency.png`: Token-by-token stepping memory (`mamba_state_bytes` flat across `256..4096` with `max/min <= 1.05`; `transformer_kv_bytes` growing linearly `16.0x` from `256` to `4096`) and per-token latency in milliseconds.
- **Length-4096 Summary**: In recurrent stepping mode, Mamba's state remains fixed at `19,456` bytes (`2 layers * (128*16 + 128*3) * 4` bytes in fp32), whereas the parameter-matched Causal Transformer's KV cache grows strictly linearly from `262,144` bytes at length 256 to `4,194,304` bytes at length 4096 (`16x`). Record the observed length-4096 task accuracies from `reports/train_eval.json` here after your GPU training run.
