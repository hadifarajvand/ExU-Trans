from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import torch

from config import CONFIG
from dataset import build_loaders, discover_dataset_root, write_splits
from model import ExUTransLite
from train import evaluate, train_one_epoch
from utils_device import setup_device


def main() -> int:
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
    model = ExUTransLite(in_ch=4, out_ch=out_channels, num_transformer_layers=CONFIG["transformer_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    measured_dir = CONFIG["MEASURED_DIR"]
    for sub in ("checkpoints", "training", "metrics", "predictions", "figures", "tables", "experiments"):
        (measured_dir / sub).mkdir(parents=True, exist_ok=True)

    epochs = int(os.getenv("EPOCHS", CONFIG.get("epochs", 1)))
    best_val = float("-inf")
    checkpoint_path = measured_dir / "checkpoints" / "best_model.pt"

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, CONFIG["label_mode"], device)
        val_summary, _, _ = evaluate(model, val_loader, CONFIG["label_mode"], device)
        (measured_dir / "training" / f"epoch_{epoch + 1}.json").write_text(
            json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "val": val_summary}, indent=2),
            encoding="utf-8",
        )
        if val_summary["dice"] > best_val:
            best_val = val_summary["dice"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_dice": best_val}, checkpoint_path)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    val_summary, _, val_metrics = evaluate(model, val_loader, CONFIG["label_mode"], device)
    test_summary, _, test_metrics = evaluate(model, test_loader, CONFIG["label_mode"], device)

    (measured_dir / "metrics" / "metrics_val.json").write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
    (measured_dir / "metrics" / "metrics_test.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")

    with (measured_dir / "metrics" / "metrics_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "dice", "iou", "precision", "recall", "f1", "hd95_px"])
        writer.writeheader()
        for case_id, metrics in test_metrics.items():
            writer.writerow({"case_id": case_id, **metrics})

    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main())
