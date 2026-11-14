"""Train MambaLM and matched CausalTransformerLM on synthetic tasks at length 256."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from src.data.induction import INDUCTION_VOCAB_SIZE, make_induction
from src.data.selective_copy import SELECTIVE_COPY_VOCAB_SIZE, make_selective_copy
from src.mamba.model import MambaLM
from src.mamba.scan import measure_scan_equivalence
from src.transformer_baseline import (
    CausalTransformerLM,
    count_trainable_params,
    find_matched_mlp_width,
)

TASK_CONFIGS = {
    "selective_copy": {
        "vocab_size": SELECTIVE_COPY_VOCAB_SIZE,
        "generator": make_selective_copy,
        "floor": 0.80,
        "early_stop_acc": 0.90,
    },
    "induction": {
        "vocab_size": INDUCTION_VOCAB_SIZE,
        "generator": make_induction,
        "floor": 0.50,
        "early_stop_acc": 0.92,
    },
}


def masked_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss restricted strictly to masked target indices."""
    vocab_size = logits.shape[-1]
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    flat_mask = loss_mask.reshape(-1)
    return (per_token_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)


def masked_token_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    """Compute token accuracy restricted strictly to masked target indices."""
    preds = logits.argmax(dim=-1)
    correct = ((preds == target_ids).float() * loss_mask).sum()
    total = loss_mask.sum().clamp_min(1.0)
    return float((correct / total).item())


def build_models_for_task(
    task: str,
    device: torch.device,
) -> tuple[MambaLM, CausalTransformerLM, int, int, int]:
    """Instantiate MambaLM and parameter-matched CausalTransformerLM."""
    vocab_size = TASK_CONFIGS[task]["vocab_size"]
    mamba = MambaLM(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        d_state=16,
        expand=2,
        conv_kernel=4,
    ).to(device)
    mamba_params = count_trainable_params(mamba)
    mlp_width = find_matched_mlp_width(
        mamba_param_count=mamba_params,
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
    )
    transformer = CausalTransformerLM(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_width=mlp_width,
    ).to(device)
    tf_params = count_trainable_params(transformer)
    return mamba, transformer, mamba_params, tf_params, mlp_width


def _induction_stage(step: int, max_steps: int) -> tuple[int, float]:
    """Return (sequence_length, lr) 5-stage curriculum for induction."""
    frac = step / max_steps
    if frac <= 0.25:
        return 16, 4e-3
    if frac <= 0.45:
        return 32, 3e-3
    if frac <= 0.65:
        return 64, 2e-3
    if frac <= 0.80:
        return 128, 1.5e-3
    return 256, 1e-3


def train_single_run(
    model_name: str,
    model: torch.nn.Module,
    task: str,
    device: torch.device,
    checkpoint_dir: Path,
    max_steps: int = 2000,
    batch_size: int = 32,
    train_length: int = 256,
    seed: int = 20261114,
    lr: float = 1e-3,
    mlp_width: int | None = None,
) -> dict[str, object]:
    """Train one model on one task at length 256 and save best checkpoint."""
    task_cfg = TASK_CONFIGS[task]
    generator = task_cfg["generator"]
    early_stop_acc = float(task_cfg["early_stop_acc"])
    vocab_size = int(task_cfg["vocab_size"])

    eff_batch = 128 if (task == "induction" and device.type == "cuda") else batch_size
    wd = 0.0 if task == "induction" else 0.01

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Fixed held-out validation batch at target length 256.
    val_seed = seed + 900000
    val_x, val_y, val_mask = generator(batch=64, length=train_length, seed=val_seed)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    val_mask = val_mask.to(device)

    best_val_acc = -1.0
    best_step = 0
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    start_time = time.perf_counter()

    micro_bs = 4 if (device.type == "cpu" and eff_batch % 4 == 0) else eff_batch
    n_micro = eff_batch // micro_bs

    steps_run = 0
    for step in range(1, max_steps + 1):
        steps_run = step
        if task == "induction":
            cur_len, cur_lr = _induction_stage(step, max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
        else:
            cur_len = train_length

        model.train()
        batch_seed = seed + step
        x_in, y_tgt, mask = generator(batch=eff_batch, length=cur_len, seed=batch_seed)
        x_in = x_in.to(device)
        y_tgt = y_tgt.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for m_idx in range(n_micro):
            sl = slice(m_idx * micro_bs, (m_idx + 1) * micro_bs)
            logits = model(x_in[sl])
            micro_loss = masked_cross_entropy(logits, y_tgt[sl], mask[sl]) / n_micro
            micro_loss.backward()
            loss_val += float(micro_loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                val_accs = []
                for v_idx in range(0, val_x.shape[0], 8):
                    v_sl = slice(v_idx, v_idx + 8)
                    val_logits = model(val_x[v_sl])
                    val_accs.append(masked_token_accuracy(val_logits, val_y[v_sl], val_mask[v_sl]))
                val_acc = sum(val_accs) / len(val_accs)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = step
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[{task} | {model_name} | L={cur_len}] step={step:4d} loss={loss_val:.4f} "
                f"val_acc_256={val_acc:.4f} best_256={best_val_acc:.4f} "
                f"elapsed={time.perf_counter()-start_time:.1f}s",
                flush=True,
            )
            if cur_len == train_length and val_acc >= early_stop_acc and step >= 100:
                break

    elapsed_s = time.perf_counter() - start_time
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{model_name}_{task}.pt"

    config = {
        "vocab_size": vocab_size,
        "d_model": 64,
        "n_layers": 2,
    }
    if model_name == "mamba":
        config.update({"d_state": 16, "expand": 2, "conv_kernel": 4})
    else:
        config.update({"n_heads": 4, "mlp_width": int(mlp_width or 256)})

    torch.save(
        {
            "model_name": model_name,
            "task": task,
            "config": config,
            "model_state_dict": best_state_dict,
            "step_count": steps_run,
            "best_step": best_step,
            "val_acc_256": best_val_acc,
            "param_count": count_trainable_params(model),
            "seed": seed,
            "lr": lr,
            "batch_size": eff_batch,
            "elapsed_seconds": elapsed_s,
        },
        ckpt_path,
    )
    return {
        "checkpoint": str(ckpt_path),
        "steps_run": steps_run,
        "best_step": best_step,
        "val_acc_256": best_val_acc,
        "elapsed_seconds": elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MambaLM and CausalTransformerLM baselines.")
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "cpu"))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "2000")))
    parser.add_argument("--tasks", nargs="+", default=["selective_copy", "induction"])
    parser.add_argument("--models", nargs="+", default=["mamba", "transformer"])
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "reports"))
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)

    scan_report = measure_scan_equivalence(out_dir / "scan_equivalence.json")
    print("Scan equivalence:", json.dumps(scan_report))

    summary: dict[str, object] = {}
    for task in args.tasks:
        torch.manual_seed(20261114)
        mamba, transformer, mamba_params, tf_params, mlp_width = build_models_for_task(task, device)
        rel_diff = abs(tf_params - mamba_params) / mamba_params
        print(
            f"Task={task}: Mamba params={mamba_params}, Transformer params={tf_params} "
            f"(mlp_width={mlp_width}, rel_diff={rel_diff:.4%})"
        )
        task_summary: dict[str, object] = {
            "mamba_params": mamba_params,
            "transformer_params": tf_params,
            "transformer_mlp_width": mlp_width,
        }
        if "mamba" in args.models:
            task_summary["mamba"] = train_single_run(
                model_name="mamba",
                model=mamba,
                task=task,
                device=device,
                checkpoint_dir=ckpt_dir,
                max_steps=args.max_steps,
            )
        if "transformer" in args.models:
            task_summary["transformer"] = train_single_run(
                model_name="transformer",
                model=transformer,
                task=task,
                device=device,
                checkpoint_dir=ckpt_dir,
                max_steps=args.max_steps,
                mlp_width=mlp_width,
            )
        summary[task] = task_summary

    print("Training complete:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
