"""Quick debug run with minimal data for testing."""

import sys
import torch
from pathlib import Path

# Add scripts directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

# Enable debug mode
from config import CONFIG
CONFIG["USE_DEBUG_SUBSET"] = True
CONFIG["DEBUG_NUM_CASES"] = 2
CONFIG["epochs"] = 2  # Just 2 epochs for quick test

from utils_device import setup_device
from dataset import summarize_dataset, build_loaders
from model import create_model
from train import train_one_epoch, evaluate
from export import setup_output_dirs


def main():
    """Quick debug training run."""
    device = setup_device()

    print("\n=== DEBUG RUN (2 epochs on 2 cases) ===\n")

    # Dataset
    summary = summarize_dataset(CONFIG)
    train_loader, val_loader, test_loader, official_val_loader = build_loaders(CONFIG, summary)

    if len(train_loader.dataset) == 0:
        print("[ERROR] No training data available.")
        sys.exit(1)

    # Model
    model = create_model(in_ch=4, out_ch=1, patch_size=CONFIG["patch_size"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    # Setup outputs
    setup_output_dirs()

    # Quick training
    print(f"Training for {CONFIG['epochs']} epochs on debug subset...")
    for epoch in range(CONFIG["epochs"]):
        tr_loss, _ = train_one_epoch(
            model, train_loader, optimizer, CONFIG["label_mode"],
            device, attr_weight=CONFIG["attr_loss_weight"]
        )
        val_metrics, _, _ = evaluate(model, val_loader, CONFIG["label_mode"], device)
        print(f"Epoch {epoch+1} - Train Loss: {tr_loss:.4f} - Val Dice: {val_metrics['dice']:.4f}")

    print("\n✓ Debug run completed successfully!")


if __name__ == "__main__":
    main()
