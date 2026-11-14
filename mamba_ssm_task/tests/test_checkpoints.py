"""Tests for Phase 6, 7 & 8: Parameter counts, reloaded checkpoints, floors, and length generalization."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval import evaluate_checkpoint, load_checkpoint_model


TASK_ROOT = Path(__file__).resolve().parent.parent


def test_param_counts() -> None:
    report_path = TASK_ROOT / "reports" / "train_eval.json"
    assert report_path.exists(), "reports/train_eval.json must exist"
    report = json.loads(report_path.read_text())

    for task in ("selective_copy", "induction"):
        task_data = report["tasks"][task]
        mamba_p = int(task_data["mamba_param_count"])
        tf_p = int(task_data["transformer_param_count"])
        mlp_width = int(task_data["transformer_mlp_width"])
        assert mlp_width > 0
        assert abs(tf_p - mamba_p) <= 0.10 * mamba_p

        mamba_model, _ = load_checkpoint_model(TASK_ROOT / "checkpoints" / f"mamba_{task}.pt", torch.device("cpu"))
        tf_model, _ = load_checkpoint_model(TASK_ROOT / "checkpoints" / f"transformer_{task}.pt", torch.device("cpu"))
        assert mamba_model.d_model == 64
        assert mamba_model.n_layers == 2
        assert tf_model.d_model == 64
        assert tf_model.n_layers == 2


def test_checkpoints_and_length_eval() -> None:
    report = json.loads((TASK_ROOT / "reports" / "train_eval.json").read_text())
    device = torch.device("cpu")

    # Check floors on >= 64 fresh sequences at length 256.
    mamba_sc, _ = load_checkpoint_model(TASK_ROOT / "checkpoints" / "mamba_selective_copy.pt", device)
    sc_acc_256 = evaluate_checkpoint(mamba_sc, "selective_copy", 256, num_sequences=64, seed=910256, device=device)
    assert sc_acc_256 >= 0.80, f"Selective copy Mamba 256 acc {sc_acc_256:.4f} < 0.80"

    mamba_ind, _ = load_checkpoint_model(TASK_ROOT / "checkpoints" / "mamba_induction.pt", device)
    ind_acc_256 = evaluate_checkpoint(mamba_ind, "induction", 256, num_sequences=64, seed=920256, device=device)
    assert ind_acc_256 >= 0.50, f"Induction Mamba 256 acc {ind_acc_256:.4f} < 0.50"

    # Verify JSON entries for all models, tasks, and lengths, and recompute 1024/4096 cells.
    for task in ("selective_copy", "induction"):
        for model_name in ("mamba", "transformer"):
            ckpt = TASK_ROOT / "checkpoints" / f"{model_name}_{task}.pt"
            assert ckpt.exists(), f"Missing checkpoint {ckpt}"
            model, _ = load_checkpoint_model(ckpt, device)
            lengths_dict = report["tasks"][task]["models"][model_name]["lengths"]
            for length_str in ("256", "1024", "4096"):
                cell = lengths_dict[length_str]
                acc = float(cell["token_accuracy"])
                assert 0.0 <= acc <= 1.0
                if length_str in ("1024", "4096"):
                    recomputed = evaluate_checkpoint(
                        model,
                        task,
                        int(length_str),
                        num_sequences=int(cell.get("num_sequences", 16)),
                        seed=int(cell["eval_seed"]),
                        device=device,
                    )
                    assert abs(recomputed - acc) <= 0.05

