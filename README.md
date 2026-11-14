# Mamba Selective SSM Workbench

A hands-on implementation and interactive explorer for **Mamba Selective State-Space Models** written from scratch in pure PyTorch (zero imports from `mamba_ssm` or custom CUDA kernels), paired with a browser-based workbench for stepping through the math by hand.

If you want to read the full conceptual walkthrough—covering zero-order-hold discretization, input-selective gating ($\Delta_t, B_t, C_t$), how associative prefix scans parallelize a sequential recurrence, why constant $O(1)$ inference state beats a growing KV cache, and what actually happened when we trained on Google Colab at $L=256$ and pushed evaluation out to $L=4096$—start with **[`mamba_ssm_task/README.md`](mamba_ssm_task/README.md)**.

---

## What Is Inside

- **`mamba_ssm_task/`** — The standalone PyTorch implementation, synthetic memory datasets (`selective_copy` and `induction`), parameter-matched `CausalTransformerLM` baseline (`~98k` params each), Colab-trained `.pt` checkpoints, evaluation reports, `pytest` suite (`20/20` passing), and Docker setup.
- **`src/`** — Interactive React + Vite + Tailwind workbench (`http://localhost:3000`) built around four live views:
  1. **Selective Discretization & Parallel Scan Lab** — Tweak $A$, $\Delta_t$, and $x_t$ live, watch sequential vs. Hillis-Steele parallel scan trees agree step by step, and verify associativity errors (`0.00e+0`).
  2. **Synthetic Memory Tasks & Length Generalization** — Inspect the actual token sequences and loss masks for Selective Copying and Induction Heads alongside the empirical `L=256 → 1024 → 4096` accuracy curves.
  3. **Recurrent Memory & Per-Token Latency Benchmark** — Compare Mamba's flat `19.0 KiB` state against the Transformer's linearly growing KV cache (`256 KiB → 4,096 KiB`) with an interactive token-by-token stepper and Matplotlib PNG exports.
  4. **PyTorch Source & Automated Verification Matrix** — Browse the core PyTorch modules (`discretize.py`, `scan.py`, `block.py`, `model.py`, `transformer_baseline.py`) next to the scan-equivalence measurements.

---

## Quick Start

### Run the PyTorch Project & Tests

```bash
cd mamba_ssm_task
pip install -r requirements.txt
pytest -v
```

### Run the Interactive Web Workbench

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`. To run the TypeScript math/scan unit checks:

```bash
npx tsx --test src/lib/mambaSim.test.ts
```
