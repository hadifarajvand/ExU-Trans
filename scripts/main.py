from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import torch

from config import CONFIG
from dataset import build_loaders, discover_dataset_root, write_splits
from model import ExUTransLite
from train import evaluate, train_one_epoch
from utils_device import setup_device


def _write_metrics_csv(path: Path, metrics_by_case: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for metrics in metrics_by_case.values() for key in metrics})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", *metric_keys])
        writer.writeheader()
        for case_id, metrics in metrics_by_case.items():
            writer.writerow({"case_id": case_id, **metrics})


def main() -> int:
    started = time.time()
    dataset_root = discover_dataset_root(CONFIG["DATASET_ROOT"])
    if dataset_root is None:
        print("ERROR: No BraTS2020 HDF5 dataset found.")
        return 1

    CONFIG["dataset_root"] = dataset_root
    write_splits(dataset_root, CONFIG["split_dir"], seed=CONFIG["RANDOM_SEED"])
    summary = {"dataset_root": dataset_root}
    train_loader, val_loader, test_loader, _ = build_loaders(CONFIG, summary)
    if len(train_loader.dataset) == 0:
        print("ERROR: No training slices found.")
        return 1

    device = setup_device()
    out_channels = 1 if CONFIG["label_mode"] == "whole_tumor" else 3
    model = ExUTransLite(
        in_ch=4,
        out_ch=out_channels,
        num_transformer_layers=CONFIG["transformer_layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    measured_dir = CONFIG["MEASURED_DIR"]
    for sub in ("checkpoints", "training", "metrics", "predictions", "figures", "tables", "experiments"):
        (measured_dir / sub).mkdir(parents=True, exist_ok=True)

    epochs = int(os.getenv("EPOCHS", CONFIG.get("epochs", 1)))
    best_val = float("-inf")
    best_path = measured_dir / "checkpoints" / "best_model.pt"
    last_path = measured_dir / "checkpoints" / "last_model.pt"

    history = []
    for epoch in range(epochs):
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
        (measured_dir / "training" / f"epoch_{epoch + 1}.json").write_text(
            json.dumps(epoch_payload, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_dice": val_summary.get("dice", float("nan")),
            },
            last_path,
        )
        if val_summary.get("dice", float("-inf")) > best_val:
            best_val = val_summary["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_dice": best_val,
                },
                best_path,
            )

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state"])
    val_summary, _, val_metrics = evaluate(model, val_loader, CONFIG["label_mode"], device)
    test_summary, _, test_metrics = evaluate(model, test_loader, CONFIG["label_mode"], device)

    (measured_dir / "metrics" / "metrics_val.json").write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
    (measured_dir / "metrics" / "metrics_test.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    _write_metrics_csv(measured_dir / "metrics" / "metrics_validation.csv", val_metrics)
    _write_metrics_csv(measured_dir / "metrics" / "metrics_test.csv", test_metrics)

    manifest = {
        "status": "MEASURED",
        "run_type": "FULL_OR_USER_CONFIGURED_REAL_DATA",
        "dataset_root": str(dataset_root),
        "dataset_redownloaded": False,
        "label_mode": CONFIG["label_mode"],
        "output_channels": out_channels,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": CONFIG["batch_size"],
        "device": str(device),
        "learning_rate": CONFIG["learning_rate"],
        "transformer_layers": CONFIG["transformer_layers"],
        "train_slices": len(train_loader.dataset),
        "validation_slices": len(val_loader.dataset),
        "test_slices": len(test_loader.dataset),
        "best_epoch": int(state.get("epoch", 0)),
        "best_validation_dice": float(state.get("val_dice", float("nan"))),
        "elapsed_seconds": time.time() - started,
        "warning": "Measured outputs are generated from this repository's actual model run; paper reference values remain separate.",
    }
    (measured_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
