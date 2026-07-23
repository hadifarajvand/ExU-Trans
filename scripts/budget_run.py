"""Time-budgeted real-data training/evaluation for genuine measured artifacts.

This is intentionally a subset experiment, not a full-paper reproduction. It uses
fixed volume-level splits, real HDF5 samples, real optimization, best-validation
checkpointing, real test metrics, and a measured-only exporter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import HDF5BratsSliceDataset, _parse_volume_slice, discover_dataset_root, write_splits
from export_measured_paper_style import export_measured_outputs
from model import ExUTransLite
from train import evaluate, train_one_epoch
from utils_device import setup_device


def _read_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_ids(ids: list[int], count: int, seed: int) -> list[int]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[: min(count, len(ids))])


def _files_for_ids(files: list[Path], ids: list[int]) -> list[Path]:
    wanted = set(ids)
    return [path for path in files if _parse_volume_slice(path)[0] in wanted]


def _write_metrics_csv(path: Path, metrics_by_case: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for metrics in metrics_by_case.values() for key in metrics})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", *metric_keys])
        writer.writeheader()
        for case_id, metrics in metrics_by_case.items():
            writer.writerow({"case_id": case_id, **metrics})


def _save_training_curve(history: list[dict], output_path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    losses = [item["train_loss"] for item in history]
    val_dice = [item["val"].get("dice", np.nan) for item in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, losses, marker="o", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_dice, marker="s", linestyle="--", label="Validation Dice")
    ax2.set_ylabel("Validation Dice")
    ax1.grid(alpha=0.3)
    fig.suptitle("Time-budgeted real-data training history")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_qualitative_preview(preview: list, output_path: Path) -> None:
    if not preview:
        return
    samples = preview[:2]
    fig, axes = plt.subplots(len(samples), 4, figsize=(13, 3.5 * len(samples)), squeeze=False)
    for row, (images, masks, pred, case_id, slice_idx, _metadata) in enumerate(samples):
        image = images[0].numpy()
        target = masks[0].numpy()
        prediction = pred[0]
        target_union = np.any(target > 0, axis=0) if target.ndim == 3 else target.squeeze() > 0
        pred_union = np.any(prediction > 0, axis=0) if prediction.ndim == 3 else prediction.squeeze() > 0
        error = np.logical_xor(target_union, pred_union)
        panels = [image[0], target_union, pred_union, error]
        titles = ["MRI channel 0", "Ground truth union", "Prediction union", "Error map"]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel, cmap="gray")
            axes[row, col].set_title(f"{title}\ncase={case_id}, slice={slice_idx}")
            axes[row, col].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plan_subset(hours: float, epochs: int, batch_size: int, seconds_per_batch: float, slices_per_volume: int = 155):
    # Reserve ~45% of wall time for repeated validation, final validation/test,
    # checkpoint I/O and figure/table export. The 6.8 s/batch default comes from
    # the measured CPU benchmark on this codebase and can be overridden.
    train_budget_seconds = hours * 3600.0 * 0.55
    train_batches_per_epoch = max(1, int(train_budget_seconds / max(seconds_per_batch * epochs, 1e-6)))
    train_slices = train_batches_per_epoch * batch_size
    train_volumes = max(4, int(train_slices // slices_per_volume))
    val_volumes = max(2, int(round(train_volumes * 0.20)))
    test_volumes = max(2, int(round(train_volumes * 0.20)))
    return train_volumes, val_volumes, test_volumes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a genuine real-data subset experiment within a wall-time budget")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--baseline-seconds-per-batch", type=float, default=6.8)
    parser.add_argument("--train-volumes", type=int, default=None)
    parser.add_argument("--val-volumes", type=int, default=None)
    parser.add_argument("--test-volumes", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    budget_seconds = args.hours * 3600.0
    dataset_root = discover_dataset_root(CONFIG["DATASET_ROOT"])
    if dataset_root is None:
        raise FileNotFoundError("Pre-downloaded BraTS2020 HDF5 dataset not found; budget run never auto-downloads data")

    # Reuse the deterministic volume-level split. write_splits is deterministic
    # and does not inspect/alter image content.
    write_splits(dataset_root, CONFIG["split_dir"], seed=CONFIG["RANDOM_SEED"])
    split_dir = CONFIG["split_dir"]
    train_ids_all = _read_ids(split_dir / "train_volumes.txt")
    val_ids_all = _read_ids(split_dir / "validation_volumes.txt")
    test_ids_all = _read_ids(split_dir / "test_volumes.txt")

    auto_train, auto_val, auto_test = _plan_subset(
        args.hours,
        args.epochs,
        CONFIG["batch_size"],
        args.baseline_seconds_per_batch,
    )
    train_count = args.train_volumes or auto_train
    val_count = args.val_volumes or auto_val
    test_count = args.test_volumes or auto_test

    train_ids = _select_ids(train_ids_all, train_count, CONFIG["RANDOM_SEED"] + 11)
    val_ids = _select_ids(val_ids_all, val_count, CONFIG["RANDOM_SEED"] + 22)
    test_ids = _select_ids(test_ids_all, test_count, CONFIG["RANDOM_SEED"] + 33)

    all_files = sorted(path for path in dataset_root.glob("*.h5") if path.is_file())
    train_files = _files_for_ids(all_files, train_ids)
    val_files = _files_for_ids(all_files, val_ids)
    test_files = _files_for_ids(all_files, test_ids)

    metadata_csv = CONFIG.get("metadata_csv")
    train_ds = HDF5BratsSliceDataset(train_files, CONFIG["image_size"], CONFIG["label_mode"], True, None, True, True, metadata_csv)
    val_ds = HDF5BratsSliceDataset(val_files, CONFIG["image_size"], CONFIG["label_mode"], False, None, False, True, metadata_csv)
    test_ds = HDF5BratsSliceDataset(test_files, CONFIG["image_size"], CONFIG["label_mode"], False, None, False, True, metadata_csv)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CONFIG["num_workers"])

    if not train_ds or not val_ds or not test_ds:
        raise RuntimeError("Budget subset resolved to an empty train/validation/test dataset")

    device = setup_device()
    out_channels = 1 if CONFIG["label_mode"] == "whole_tumor" else 3
    model = ExUTransLite(in_ch=4, out_ch=out_channels, num_transformer_layers=CONFIG["transformer_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    minutes = int(round(args.hours * 60))
    output_root = CONFIG["OUTPUT_ROOT"] / f"measured_budget_{minutes}min"
    for sub in ("checkpoints", "training", "metrics", "figures", "tables"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    plan = {
        "status": "PLANNED_REAL_SUBSET",
        "time_budget_hours": args.hours,
        "baseline_seconds_per_batch": args.baseline_seconds_per_batch,
        "epochs_requested": args.epochs,
        "train_volume_ids": train_ids,
        "validation_volume_ids": val_ids,
        "test_volume_ids": test_ids,
        "train_slices": len(train_ds),
        "validation_slices": len(val_ds),
        "test_slices": len(test_ds),
        "batch_size": CONFIG["batch_size"],
        "estimated_train_batches_per_epoch": math.ceil(len(train_ds) / CONFIG["batch_size"]),
        "scope": "REAL_DATA_SUBSET_NOT_FULL_PAPER_REPRODUCTION",
    }
    (output_root / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    history = []
    best_val = float("-inf")
    best_path = output_root / "checkpoints" / "best_model.pt"
    last_path = output_root / "checkpoints" / "last_model.pt"

    for epoch in range(args.epochs):
        # Leave time for final validation/test and exports. We only stop between
        # complete epochs so every reported epoch is scientifically interpretable.
        if epoch > 0 and (time.time() - started) >= budget_seconds * 0.70:
            break
        epoch_started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG["label_mode"], device)
        val_summary, _, _ = evaluate(model, val_loader, CONFIG["label_mode"], device)
        epoch_payload = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val": val_summary,
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(epoch_payload)
        (output_root / "training" / f"epoch_{epoch + 1}.json").write_text(json.dumps(epoch_payload, indent=2), encoding="utf-8")
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch + 1, "val_dice": val_summary.get("dice")}, last_path)
        if val_summary.get("dice", float("-inf")) > best_val:
            best_val = val_summary["dice"]
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch + 1, "val_dice": best_val}, best_path)

    if not best_path.exists():
        raise RuntimeError("No completed epoch/checkpoint available for final evaluation")

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state"])
    val_summary, _, val_metrics = evaluate(model, val_loader, CONFIG["label_mode"], device)
    test_summary, test_preview, test_metrics = evaluate(model, test_loader, CONFIG["label_mode"], device)

    metrics_dir = output_root / "metrics"
    (metrics_dir / "metrics_validation.json").write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
    (metrics_dir / "metrics_test.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    _write_metrics_csv(metrics_dir / "metrics_validation.csv", val_metrics)
    test_csv = metrics_dir / "metrics_test.csv"
    _write_metrics_csv(test_csv, test_metrics)

    _save_training_curve(history, output_root / "figures" / "figure_training_history.png")
    _save_qualitative_preview(test_preview, output_root / "figures" / "figure_real_qualitative_examples.png")
    export_manifest = export_measured_outputs(test_csv, output_root)

    manifest = {
        "status": "MEASURED_BUDGET_RUN_COMPLETE",
        "scope": "REAL_DATA_SUBSET_NOT_FULL_PAPER_REPRODUCTION",
        "dataset_root": str(dataset_root),
        "dataset_redownloaded": False,
        "time_budget_hours": args.hours,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "label_mode": CONFIG["label_mode"],
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": int(state.get("epoch", 0)),
        "best_validation_dice": float(state.get("val_dice", float("nan"))),
        "train_volume_ids": train_ids,
        "validation_volume_ids": val_ids,
        "test_volume_ids": test_ids,
        "train_slices": len(train_ds),
        "validation_slices": len(val_ds),
        "test_slices": len(test_ds),
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "transformer_layers": CONFIG["transformer_layers"],
        "test_summary": test_summary,
        "export_manifest": export_manifest,
        "scientific_notes": [
            "All measured figures/tables trace to this real subset run.",
            "No synthetic fallback or hard-coded measured values are used.",
            "Region channel names remain Region_0/1/2 unless provenance is proven.",
            "Subset metrics are not claimed as the full-paper result.",
        ],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
