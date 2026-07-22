"""Main training script for ExU-Trans BraTS2020 reproduction."""

import sys
import torch
from pathlib import Path

# Add scripts directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG
from utils_device import setup_device
from dataset import summarize_dataset, build_loaders
from model import create_model
from train import train_one_epoch, evaluate
from export import setup_output_dirs, export_metrics_table, export_comparison_chart
from visualize import show_sample


def verify_dataset(train_loader, config):
    """Verify dataset is loaded correctly."""
    if len(train_loader.dataset) == 0:
        print("[ERROR] No training data available. Check dataset paths.")
        return False

    sample = train_loader.dataset[0]
    image, mask, case_id, slice_idx = sample
    print('Dataset verification:')
    print('  Case:', case_id)
    print('  Image shape:', image.shape)
    print('  Mask shape:', mask.shape)
    print('  OK - Labeled training case loaded successfully')
    return True


def verify_model(model, train_loader, device, config):
    """Verify model forward pass."""
    if len(train_loader.dataset) == 0:
        return False

    images, masks, case_id, slice_idx = next(iter(train_loader))
    images = images[:1].to(device)
    with torch.no_grad():
        logits, aux = model(images)

    print(f"\nModel forward pass verification:")
    print(f"  Input shape: {tuple(images.shape)} (B, C, H, W)")
    print(f"  Logits shape: {tuple(logits.shape)}")
    print(f"  Attr map shape: {tuple(aux['attr_map'].shape)}")
    print(f"  OK - Model forward pass works")
    return True


def main():
    """Main training loop."""
    # Setup device and seed
    device = setup_device()

    # Dataset discovery and loading
    print("\n=== DATASET PREPARATION ===")
    summary = summarize_dataset(CONFIG)
    train_loader, val_loader, test_loader, official_val_loader = build_loaders(CONFIG, summary)

    # Verify dataset
    if not verify_dataset(train_loader, CONFIG):
        sys.exit(1)

    # Setup output directories
    setup_output_dirs()

    # Create model
    print("\n=== MODEL SETUP ===")
    out_ch = 1 if CONFIG["label_mode"] == "whole_tumor" else 4
    model = create_model(in_ch=4, out_ch=out_ch, patch_size=CONFIG["patch_size"], device=device)
    print(f"Model: {model.__class__.__name__}")

    # Verify model
    if not verify_model(model, train_loader, device, CONFIG):
        print("[WARNING] Model verification skipped - no data available")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    print(f"Optimizer: AdamW (lr={CONFIG['lr']}, weight_decay={CONFIG['weight_decay']})")

    # Training
    if len(train_loader.dataset) > 0:
        print(f"\n=== TRAINING ===")
        print(f"Starting training: {CONFIG['epochs']} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        best_val_dice = 0.0
        train_losses = []
        val_scores = []

        for epoch in range(CONFIG["epochs"]):
            tr_loss, loss_components = train_one_epoch(
                model, train_loader, optimizer, CONFIG["label_mode"],
                device, attr_weight=CONFIG["attr_loss_weight"],
                align_weight=0.05, boundary_weight=0.05
            )
            train_losses.append(tr_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metrics, val_samples, val_case_metrics = evaluate(model, val_loader, CONFIG["label_mode"], device)
                val_scores.append(val_metrics)

                if val_metrics["dice"] > best_val_dice:
                    best_val_dice = val_metrics["dice"]

                print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train loss: {tr_loss:.4f} - Val Dice: {val_metrics['dice']:.4f}")

        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Best Val Dice: {best_val_dice:.4f}")

        # Export validation metrics
        print(f"\n=== EXPORTING RESULTS ===")
        export_metrics_table(val_case_metrics, "validation")
        export_comparison_chart(list(val_case_metrics.values()))

        print("\n=== SUMMARY ===")
        print("ExU-Trans BraTS2020 reproduction (2D slice-based):")
        print("  - Labeled training (369 cases) split: 70% train, 15% val, 15% test")
        print("  - Multi-modal (FLAIR, T1, T1ce, T2) with z-score normalization")
        print("  - Dual-encoder: U-Net + ViT with SE-MHA, DAE, contextual self-attention, bivariate fusion")
        print("  - TGOF loss: BCE + Dice + alignment + boundary-aware")
        print("  - Metrics: Dice, IoU, Precision, Recall, F1, HD95")

    else:
        print("\nTraining skipped (debug mode or no data).")


if __name__ == "__main__":
    main()
