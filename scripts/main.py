"""Measured training path for the ExU-Trans approximation."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG, CHECKPOINT_DIR
from utils_device import setup_device
from dataset import summarize_dataset, build_loaders
from model import create_model
from train import train_one_epoch, evaluate
from export import (
    setup_output_dirs, export_metrics_table,
    export_comparison_chart, write_run_manifest,
)


def main() -> int:
    device = setup_device()
    setup_output_dirs()
    summary = summarize_dataset(CONFIG)
    if summary["train_count"] == 0:
        print("ERROR: No BraTS2020 training cases found.")
        print("Set DATA_ROOT or DATA_ROOT_TRAIN. For a data-free check run: python run_full_pipeline.py smoke")
        return 2

    train_loader, val_loader, test_loader, _ = build_loaders(CONFIG, summary)
    if len(train_loader.dataset) == 0:
        print("ERROR: Training dataset contains no usable labeled slices.")
        return 2

    out_ch = 1 if CONFIG["label_mode"] == "whole_tumor" else 4
    model = create_model(
        in_ch=4,
        out_ch=out_ch,
        patch_size=CONFIG["patch_size"],
        device=str(device),
        num_transformer_layers=CONFIG["num_transformer_layers"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    best_val_dice = float("-inf")
    best_state = None
    best_epoch = None

    for epoch in range(CONFIG["epochs"]):
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, CONFIG["label_mode"], device,
            pixel_weight=CONFIG["pixel_loss_weight"],
            align_weight=CONFIG["align_loss_weight"],
            boundary_weight=CONFIG["boundary_loss_weight"],
        )
        should_eval = (
            epoch == 0
            or (epoch + 1) % CONFIG["eval_every"] == 0
            or epoch + 1 == CONFIG["epochs"]
        )
        if not should_eval:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} loss={train_loss:.5f}")
            continue

        val_summary, _, _ = evaluate(model, val_loader, CONFIG["label_mode"], device)
        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']} "
            f"loss={train_loss:.5f} val_dice={val_summary['dice']:.5f}"
        )
        if val_summary["dice"] > best_val_dice:
            best_val_dice = val_summary["dice"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        print("ERROR: No valid checkpoint candidate was produced.")
        return 3

    model.load_state_dict(best_state)
    if CONFIG["SAVE_CHECKPOINTS"]:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "config": CONFIG,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
        }, CHECKPOINT_DIR / "best_model.pt")

    val_summary, _, val_metrics = evaluate(model, val_loader, CONFIG["label_mode"], device)
    test_summary, _, test_metrics = evaluate(model, test_loader, CONFIG["label_mode"], device)
    export_metrics_table(val_metrics, "validation")
    export_metrics_table(test_metrics, "test")
    export_comparison_chart(list(val_metrics.values()), "validation")
    export_comparison_chart(list(test_metrics.values()), "test")

    manifest = {
        "implementation_class": "ExUTransLite research approximation",
        "paper_exact_reproduction": False,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "validation_summary": val_summary,
        "test_summary": test_summary,
        "case_counts": {k: v for k, v in summary.items() if k.endswith("_count")},
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "config": CONFIG,
        "known_nonpaper_assumptions": [
            "70/15/15 split because exact paper split IDs are not published",
            "128x128 default resize because exact paper resize resolution is not published",
            "lightweight channel widths are repository choices; paper reports 50.3M parameters",
            "HD95 is exported as grid pixels, not millimetres, because physical spacing after resize is not calibrated"
        ],
    }
    write_run_manifest(manifest)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
