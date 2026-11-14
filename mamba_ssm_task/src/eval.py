"""Evaluate trained Mamba and Transformer checkpoints at lengths 256, 1024, and 4096."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.data.induction import make_induction
from src.data.selective_copy import make_selective_copy
from src.mamba.model import MambaLM
from src.train import masked_token_accuracy
from src.transformer_baseline import CausalTransformerLM, count_trainable_params

EVAL_LENGTHS = [256, 1024, 4096]
EVAL_BATCH_COUNTS = {256: 64, 1024: 32, 4096: 32}
TASK_GENERATORS = {
    "selective_copy": make_selective_copy,
    "induction": make_induction,
}


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    """Instantiate model from saved checkpoint metadata and load weights."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_name = payload["model_name"]
    cfg = payload["config"]
    if model_name == "mamba":
        model: torch.nn.Module = MambaLM(
            vocab_size=int(cfg["vocab_size"]),
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            d_state=int(cfg["d_state"]),
            expand=int(cfg["expand"]),
            conv_kernel=int(cfg["conv_kernel"]),
        )
    else:
        model = CausalTransformerLM(
            vocab_size=int(cfg["vocab_size"]),
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            mlp_width=int(cfg["mlp_width"]),
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def evaluate_checkpoint(
    model: torch.nn.Module,
    task: str,
    length: int,
    num_sequences: int,
    seed: int,
    device: torch.device,
) -> float:
    """Compute masked token accuracy on fresh sequences of the requested length."""
    generator = TASK_GENERATORS[task]
    batch_size = min(16, num_sequences)
    total_correct = 0.0
    total_masked = 0.0
    n_batches = (num_sequences + batch_size - 1) // batch_size

    with torch.no_grad():
        for b_idx in range(n_batches):
            cur_bs = min(batch_size, num_sequences - b_idx * batch_size)
            x_in, y_tgt, mask = generator(batch=cur_bs, length=length, seed=seed + b_idx * 97)
            x_in = x_in.to(device)
            y_tgt = y_tgt.to(device)
            mask = mask.to(device)
            logits = model(x_in)
            preds = logits.argmax(dim=-1)
            total_correct += float(((preds == y_tgt).float() * mask).sum().item())
            total_masked += float(mask.sum().item())

    return float(total_correct / max(total_masked, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints across lengths 256, 1024, 4096.")
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "cpu"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "reports"))
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "train_length": 256,
        "eval_lengths": EVAL_LENGTHS,
        "tasks": {},
    }

    for task in ("selective_copy", "induction"):
        mamba_model, mamba_meta = load_checkpoint_model(ckpt_dir / f"mamba_{task}.pt", device)
        tf_model, tf_meta = load_checkpoint_model(ckpt_dir / f"transformer_{task}.pt", device)

        mamba_params = count_trainable_params(mamba_model)
        tf_params = count_trainable_params(tf_model)
        mlp_width = int(tf_meta["config"]["mlp_width"])

        task_entry: dict[str, object] = {
            "mamba_param_count": mamba_params,
            "transformer_param_count": tf_params,
            "transformer_mlp_width": mlp_width,
            "param_relative_diff": abs(tf_params - mamba_params) / mamba_params,
            "mamba_step_count": int(mamba_meta["step_count"]),
            "transformer_step_count": int(tf_meta["step_count"]),
            "models": {
                "mamba": {"step_count": int(mamba_meta["step_count"]), "lengths": {}},
                "transformer": {"step_count": int(tf_meta["step_count"]), "lengths": {}},
            },
        }

        for length in EVAL_LENGTHS:
            n_seq = EVAL_BATCH_COUNTS[length]
            eval_seed = 777000 + length + (0 if task == "selective_copy" else 10000)
            mamba_acc = evaluate_checkpoint(mamba_model, task, length, n_seq, eval_seed, device)
            tf_acc = evaluate_checkpoint(tf_model, task, length, n_seq, eval_seed, device)
            task_entry["models"]["mamba"]["lengths"][str(length)] = {
                "token_accuracy": round(mamba_acc, 6),
                "num_sequences": n_seq,
                "eval_seed": eval_seed,
            }
            task_entry["models"]["transformer"]["lengths"][str(length)] = {
                "token_accuracy": round(tf_acc, 6),
                "num_sequences": n_seq,
                "eval_seed": eval_seed,
            }
            print(f"[{task} | L={length:4d}] Mamba={mamba_acc:.4f} | Transformer={tf_acc:.4f}")

        report["tasks"][task] = task_entry

    out_path = out_dir / "train_eval.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote evaluation report to {out_path}")


if __name__ == "__main__":
    main()
