# Mamba From Scratch in Pure PyTorch

Most explanations of Mamba either stay at 30,000 feet ("it's linear-time attention!") or dive straight into fused CUDA kernels that are painful to read. This project takes the middle road: every piece of a selective state-space model—zero-order-hold discretization, sequential recurrence, parallel associative prefix scan, custom backward adjoint, causal depthwise 1D convolution, gated blocks, and token-by-token recurrent inference—is written from scratch in readable PyTorch, with zero imports from `mamba_ssm` or vendor CUDA extensions.

To see whether the theory actually holds up, we also build a parameter-matched 2-layer Causal Transformer (`~98k` parameters each, within `0.16%`) and pit them against each other on two classic synthetic memory tasks, then watch what happens when we stretch sequence lengths from `256` tokens out to `4,096`.

---

## The Core Puzzle: Fast Training vs. Cheap Generation

Before looking at equations, it helps to notice the awkward trade-off that shaped sequence modeling for years:

- **RNNs** are wonderfully cheap when generating text one token at a time. They compress everything they have read so far into one fixed-size hidden state $h_t$, so token #4,096 costs the exact same memory and compute as token #10. The catch? During training, step $t$ waits for step $t-1$, which wastes GPU parallelism.
- **Transformers** flip that trade-off. During training, self-attention processes all $L$ tokens in parallel. During generation, however, attention has to look back at every past Key and Value vector, so its KV cache grows linearly with sequence length ($O(L)$ memory, $O(L)$ dot-product work per new token).

**Mamba's trick** is to run a recurrence that looks linear enough in time to be parallelized via an **associative scan** during training, while remaining **input-selective** so it can decide what to remember and what to ignore—and still collapse back down to an $O(1)$ recurrent state at inference time.

---

## Key Concepts Explained (With Exact Code Mapping)

### 1. Continuous State Spaces & Zero-Order-Hold Discretization (`src/mamba/discretize.py`)

A continuous-time diagonal state-space model tracks a hidden memory $h(t)$ driven by a scalar input $x(t)$:

$$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t)$$

Think of $h(t)$ like a leaky water tank:
- $A < 0$ controls the leak rate (more negative means older memories fade faster; constraining $A = -\exp(A_{\log})$ guarantees stability so hidden states don't blow up).
- $B$ controls how strongly the incoming signal fills the tank.
- $\Delta > 0$ is the **step size**—how long the continuous system runs before sampling the next token.

Computers process discrete tokens $x_0, x_1, \dots, x_{L-1}$, not continuous curves. Under **Zero-Order Hold (ZOH)**, we treat $x(t)$ as constant over each interval of width $\Delta$ and integrate the ODE exactly in closed form:

$$\bar{A} = \exp(\Delta A), \qquad \bar{B} = \frac{\exp(\Delta A) - 1}{A}\,B$$

One subtle numerical detail in `discretize.py`: when $\Delta A$ is tiny, computing `torch.exp(delta * A) - 1` directly loses floating-point precision through catastrophic cancellation. PyTorch's `torch.expm1(delta * A)` avoids that loss cleanly.

**Sanity check you can do in your head:** plug in $A = -1$, $\Delta = \ln 2 \approx 0.6931$, $B = 2$, $C = 1$ with inputs $x = [3, 1]$:
- $\bar{A} = e^{-\ln 2} = 0.5$
- $\bar{B} = \frac{0.5 - 1}{-1} \cdot 2 = 1.0$
- Step $t=0$: $h_0 = 0.5(0) + 1.0(3) = 3.0$
- Step $t=1$: $h_1 = 0.5(3.0) + 1.0(1) = 2.5$

Because $\Delta > 0$ and $A < 0$, $\bar{A}$ always sits strictly inside $(0, 1)$—contractive, never exploding.

---

### 2. Why "Selective" Matters: Making $\Delta, B, C$ Depend on the Input

Classic state-space models (like S4) keep $\Delta$, $B$, and $C$ fixed across time. That works surprisingly well on smooth audio or continuous signals, because every step is equally informative.

Language and symbolic memory tasks are the opposite: 90% of a sequence might be filler, punctuated by a handful of tokens you *must* keep for 200 steps. If $\bar{A}$ and $\bar{B}$ are constant, the model treats noise and signal identically—noise gradually washes out the signal.

Mamba makes three parameters **functions of the current token representation** $x_t$ (inside `src/mamba/block.py`):
- $\Delta_t = \text{softplus}(\text{Linear}(x_t))$ — the **memory gate**:
  - **Big $\Delta_t$**: $\bar{A}_t = \exp(\Delta_t A)$ shrinks toward $0$ and $\bar{B}_t$ grows, so the state resets/overwrites with the new token.
  - **Tiny $\Delta_t \to 0$**: $\bar{A}_t \to 1$ and $\bar{B}_t \to 0$, so $h_t \approx h_{t-1}$—the model coasts right past filler tokens without corrupting what it is already holding.
- $B_t = \text{Linear}_B(x_t)$ and $C_t = \text{Linear}_C(x_t)$ — let each token decide *which* latent state dimensions (`d_state = 16`) it writes to and reads from.

---

### 3. Training Without a Loop: The Parallel Associative Scan (`src/mamba/scan.py`)

Once $\bar{A}_t$ and $\bar{B}_t$ vary at every timestep, we can no longer turn the recurrence into a global FFT convolution (as S4 did). At first glance, $h_t = a_t h_{t-1} + b_t$ (where $a_t = \bar{A}_t$ and $b_t = \bar{B}_t x_t$) looks stubbornly sequential.

Here is the algebraic rescue: **1D affine functions compose associatively**.

Represent each timestep $t$ as the pair $(a_t, b_t)$, meaning "multiply the incoming state by $a_t$ and add $b_t$." If an earlier span transforms state by $(a_1, b_1)$ and a later span transforms it by $(a_2, b_2)$, running one after the other gives:

$$a_2(a_1 h + b_1) + b_2 = (a_2 a_1)\,h + (a_2 b_1 + b_2)$$

So we define:

$$\text{compose}\big((a_2, b_2),\,(a_1, b_1)\big) = (a_2 a_1,\; a_2 b_1 + b_2)$$

Because $(p_3 \circ p_2) \circ p_1 = p_3 \circ (p_2 \circ p_1)$, we can regroup the additions and multiplications into a balanced binary tree (depth $O(\log L)$) instead of a chain of length $L$!

**Concrete 4-step trace** with $a = [0.5, 0.5, 0.25, 0.5]$ and $b = [1, 1, 1, 1]$:
- Sequential loop: $h_0 = 1.0 \;\to\; h_1 = 0.5(1.0)+1 = 1.5 \;\to\; h_2 = 0.25(1.5)+1 = 1.375 \;\to\; h_3 = 0.5(1.375)+1 = 1.6875$.
- Parallel Blelloch scan (`associative_scan`): pairs `(0,1)` and `(2,3)` reduce simultaneously, recurse, and down-sweep in $\log_2(4) = 2$ parallel rounds—producing the exact same `[1.0, 1.5, 1.375, 1.6875]`.

In `scan.py`, `_chunked_scan_fwd` runs prefix scans inside fixed-size chunks (`chunk_size=16`) and propagates carries across chunk boundaries with `associative_scan`. And for backpropagation, `_AssociativeScanFn.backward` doesn't store $O(\log L)$ intermediate tree levels—it simply runs the same chunked scan **in reverse time** on $\delta h_t = a_{t+1}\,\delta h_{t+1} + g_t$.

Empirical agreement against the naive `for`-loop (`reports/scan_equivalence.json`):
- Sequence length `8`: max absolute error `1.19e-7` (tolerance `1e-5`)
- Sequence length `128`: max absolute error `1.79e-7` (tolerance `1e-3`)

---

### 4. Anatomy of a Mamba Block & Constant-State Stepping (`src/mamba/block.py`, `src/mamba/model.py`)

A full `MambaBlock` (`d_model=64`, `expand=2` $\implies$ `d_inner=128`, `d_state=16`, `conv_kernel=4`) splits its input projection into two parallel streams:

```
x (B, L, 64)
  │
  ├──▶ in_proj ──▶ [ x_seq (128) | gate_branch (128) ]
  │                    │                 │
  │         causal Conv1d (k=4)          │
  │                    │                 │
  │                  SiLU                │
  │                    │                 │
  │          discretize + scan           │
  │                    │                 │
  │                  y_ssm ──────×────── SiLU
  │                              │
  └────────────────────────▶ out_proj (no bias) ──▶ output (B, L, 64)
```

Three design details worth noticing in `block.py`:
1. **Strict causality in Conv1d**: `nn.Conv1d(..., padding=conv_kernel - 1)` pads `3` zeros on the left and we slice `[:, :, :length]`, so token $t$ only sees `[t-3, t-2, t-1, t]`—never $t+1$.
2. **Multiplicative gating without skip bias**: `out_proj` has `bias=False` (and there is no direct $D x$ residual inside the block), so zeroing the gate weights makes the block output exactly zero—a clean invariant verified by `test_zeroing_gate_zeros_output`.
3. **Why generation memory never grows**: During `step(token)`, each layer needs only two fixed tensors:
   - SSM hidden state `h`: shape `(1, 128, 16)`
   - Convolution tail (last `k - 1 = 3` vectors): shape `(1, 128, 3)`
   Across 2 layers in `float32`, that totals $2 \times (128 \times 16 + 128 \times 3) \times 4 = \mathbf{19,456\text{ bytes}}$ (`19.0 KiB`), whether you are on token 50 or token 50,000.

---

### 5. A Fair Baseline: Parameter-Matched Causal Transformer (`src/transformer_baseline.py`)

Comparing a 2-layer Mamba against an arbitrarily sized Transformer wouldn't tell us much. `find_matched_mlp_width` searches Transformer MLP widths until total trainable parameters match `MambaLM` as closely as possible:

| Task | Vocab | MambaLM Params | CausalTransformerLM Params (`mlp_width=240`, 4 heads) | Difference |
| --- | --- | --- | --- | --- |
| Selective Copying | 18 | `98,432` | `98,272` | `0.16%` |
| Induction Heads | 16 | `98,176` | `98,016` | `0.16%` |

Unlike Mamba's constant `19,456` bytes, our `TransformerKVCache` stores `K` and `V` of shape `(1, 4, t, 16)` per layer—adding $2 \times 2 \times 4 \times 16 \times 4 = \mathbf{1,024\text{ bytes}}$ for every single generated token.

---

## Experiments & Honest Results

Both models were trained **only at sequence length $L = 256$** (`AdamW`, `lr = 1e-3`, `batch_size = 32`, `seed = 20261114`) on a Google Colab CUDA GPU, then evaluated **zero-shot** on fresh sequences at `256`, `1024` ($4\times$), and `4096` ($16\times$).

### The Two Tasks
- **Selective Copying (`src/data/selective_copy.py`)**: 16 signal tokens (`1..16`) are scattered randomly across noise (`0`) in the prefix; the last 16 positions contain recall markers (`17`), and loss is masked strictly to those last 16 targets. To succeed, the model must ignore noise and keep an ordered queue of 16 tokens alive.
- **Induction Heads (`src/data/induction.py`)**: Four `[key, value]` bigrams appear in the first half; the final token repeats one of those keys, and the model must recall the matching value at that final index.

### Zero-Shot Length Generalization (`reports/train_eval.json`)

| Task | Model | Steps | `L = 256` (train) | `L = 1024` ($4\times$) | `L = 4096` ($16\times$) |
| --- | --- | --- | --- | --- | --- |
| **Selective Copying** | **MambaLM** | `4,000` | `87.6%` (pass $\ge 80\%$) | **`19.9%`** | `5.7%` |
| **Selective Copying** | Causal Transformer | `1,300` | **`90.7%`** | `8.0%` | `6.4%` |
| **Induction Heads** | **MambaLM** | `3,225` | **`100.0%`** (pass $\ge 50\%$) | **`40.6%`** | `3.1%` |
| **Induction Heads** | Causal Transformer | `3,250` | `89.1%` | `28.1%` | `18.8%` |

### Autoregressive Memory & Per-Token Latency on CUDA (`reports/inference.json`)

| Sequence Length | Mamba State Memory | Transformer KV Cache | Mamba Latency | Transformer Latency |
| --- | --- | --- | --- | --- |
| `L = 256` | `19,456 B` (`19.0 KiB`) | `262,144 B` (`256 KiB`) | `1.273 ms/tok` | `1.135 ms/tok` |
| `L = 1024` | `19,456 B` (`19.0 KiB`) | `1,048,576 B` (`1,024 KiB`) | `1.237 ms/tok` | `1.169 ms/tok` |
| `L = 4096` | **`19,456 B` (`19.0 KiB`)** | **`4,194,304 B` (`4,096 KiB`)** | `1.436 ms/tok` | `1.189 ms/tok` |

### What These Numbers Actually Tell Us (Especially at `L = 4096`)

1. **Both models clear `L=256`, and Mamba wins `4x` extrapolation (`L=1024`)**: At `1024` tokens, Mamba retains more than double the accuracy of the Transformer on Selective Copying (`19.9%` vs. `8.0%`) and leads clearly on Induction Heads (`40.6%` vs. `28.1%`), because its recurrence doesn't depend on sinusoidal position frequencies it has never seen.
2. **Neither model generalizes zero-shot to `16x` length (`L=4096`) when trained only at `L=256`**: By 4,096 tokens, both drop toward random guessing (`~6%` across 16 copy classes). Why?
   - For the **Transformer**, sinusoidal positional embeddings beyond index 256 create unfamiliar relative phases, and softmax attention dilutes weights across $16\times$ more distractor keys.
   - For **Mamba**, training exclusively on 256-token gaps gives the optimizer no pressure to learn $\Delta_t \approx 0$ down to $10^{-4}$ precision; even a tiny learned residual decay like $\bar{A} = 0.998$ per noise token compounds to $0.998^{3800} \approx 0.0005$ over 4,000 tokens. True $16\times$ extrapolation typically requires multi-length training curricula or truncated BPTT across longer horizons.
3. **The memory curve is unmistakable**: At `L=4096`, Mamba's per-step state stays flat at **19.0 KiB** (ratio `1.00`), while the Transformer's KV cache hits **4,096 KiB** (`16.0x` growth from `L=256`, **`215.6x` larger** than Mamba's state)—matching `reports/inference_memory.png`. (In pure Python stepping on a tiny 2-layer model, per-token CUDA kernel launch overhead dominates arithmetic cost, so latency stays near `~1.2–1.4 ms/tok` for both until sequences become much longer or models wider.)

---

## Project Layout

```text
mamba_ssm_task/
├── src/
│   ├── mamba/
│   │   ├── discretize.py        # ZOH discretization (exp, expm1)
│   │   ├── scan.py              # sequential_scan, compose, Blelloch + chunked parallel_scan
│   │   ├── block.py             # MambaBlock (causal Conv1d, SiLU, selective SSM, step)
│   │   └── model.py             # MambaLM, MambaState, step_mamba
│   ├── data/
│   │   ├── selective_copy.py    # Deterministic selective copying generator
│   │   └── induction.py         # Deterministic induction-head generator
│   ├── transformer_baseline.py  # CausalTransformerLM, TransformerKVCache, param matcher
│   ├── train.py                 # Training loop (AdamW, gradient accumulation on CPU)
│   ├── eval.py                  # Multi-length evaluation (256, 1024, 4096)
│   └── bench_infer.py           # State/KV memory & latency benchmarks + PNG plots
├── tests/                       # 20 pytest checks across math, causality, checkpoints & memory
├── checkpoints/                 # Colab-trained .pt weights (Mamba & Transformer x 2 tasks)
├── reports/                     # scan_equivalence.json, train_eval.json, inference.json & PNGs
├── Dockerfile
└── docker-compose.yml
```

---

## Running It Yourself

```bash
pip install -r requirements.txt

# 1. Verify ZOH discretization, scan equivalence, block causality, and task generators
pytest tests/test_discretize.py tests/test_scan.py tests/test_block.py tests/test_tasks.py -v

# 2. Train both models at L=256 (CUDA recommended)
TRAIN_DEVICE=cuda MAX_STEPS=4000 python -m src.train

# 3. Evaluate at lengths 256, 1024, and 4096
TRAIN_DEVICE=cuda python -m src.eval

# 4. Benchmark constant state vs. growing KV cache and regenerate PNG plots
TRAIN_DEVICE=cuda python -m src.bench_infer

# 5. Run the entire 20-test suite (including checkpoint & inference checks)
pytest -v
```

Or run the verification suite inside Docker:

```bash
docker compose up --build
```
