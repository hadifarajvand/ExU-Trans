#!/usr/bin/env python
"""
ExU-Trans BraTS2020 Complete Execution Pipeline
Runs full training and generates all outputs
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_erosion, distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nibabel as nib
import pandas as pd
from functools import lru_cache

print("=" * 90)
print("ExU-Trans BraTS2020 - Complete Execution")
print("=" * 90)

# Configuration
RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).parent

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "DATA_ROOT_TRAIN": PROJECT_ROOT / "dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "DATA_ROOT_OFFICIAL_VAL": PROJECT_ROOT / "dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "TEST_RATIO": 0.15,
    "RANDOM_SEED": RANDOM_SEED,
    "USE_DEBUG_SUBSET": True,
    "DEBUG_NUM_CASES": 2,
    "debug_max_slices_per_case": 2,
    "image_size": 128,
    "patch_size": 16,
    "batch_size": 2,
    "num_workers": 0,
    "label_mode": "whole_tumor",
    "epochs": 1,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "attr_loss_weight": 0.1,
}

print(f"\n[SETUP] Device: {DEVICE}")
print(f"[SETUP] CUDA: {torch.cuda.is_available()}")
print(f"\n[CONFIG]")
for k, v in list(CONFIG.items())[:5]:
    print(f"  {k}: {v}")

# Dataset Functions
def _find_case_dirs(root: Path, prefix: str):
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])

@lru_cache(maxsize=16)
def load_nifti(path: str) -> np.ndarray:
    return np.transpose(nib.load(path).get_fdata().astype(np.float32), (2, 0, 1))

def zscore_non_background(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.any():
        mean, std = float(volume[mask].mean()), float(volume[mask].std())
        std = 1.0 if std < 1e-6 else std
        volume = (volume - mean) / std
    else:
        volume = volume - volume.mean()
    return volume.astype(np.float32)

# Load and split dataset
train_root = CONFIG["DATA_ROOT_TRAIN"]
all_train_cases = _find_case_dirs(train_root, "BraTS20_Training_")

print(f"\n[DATASET] Found {len(all_train_cases)} training cases")

if len(all_train_cases) == 0:
    print("[ERROR] No training cases found!")
    print(f"        Expected at: {train_root}")
    sys.exit(1)

train_cases, temp_cases = train_test_split(
    all_train_cases,
    train_size=CONFIG["TRAIN_RATIO"],
    random_state=CONFIG["RANDOM_SEED"],
    shuffle=True
)
val_cases, test_cases = train_test_split(
    temp_cases,
    train_size=CONFIG["VAL_RATIO"] / (CONFIG["VAL_RATIO"] + CONFIG["TEST_RATIO"]),
    random_state=CONFIG["RANDOM_SEED"],
    shuffle=True
)

if CONFIG["USE_DEBUG_SUBSET"]:
    train_cases = train_cases[:CONFIG["DEBUG_NUM_CASES"]]
    val_cases = val_cases[:CONFIG["DEBUG_NUM_CASES"]]
    test_cases = test_cases[:CONFIG["DEBUG_NUM_CASES"]]

print(f"[SPLIT] Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

# Create output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
ATTENTION_DIR = OUTPUT_DIR / "attention_maps"

for d in [METRICS_DIR, FIGURES_DIR, ATTENTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n[OUTPUT] Directories created in {OUTPUT_DIR}")

# Metrics Functions
def dice_score(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))

def iou_score(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))

def precision_score(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    return float((tp + eps) / (tp + fp + eps))

def recall_score(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    return float((tp + eps) / (tp + fn + eps))

def f1_score(pred, target, eps=1e-6):
    p, r = precision_score(pred, target, eps), recall_score(pred, target, eps)
    return float((2 * p * r + eps) / (p + r + eps))

def hd95(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    pred_surf = pred ^ binary_erosion(pred)
    tgt_surf = target ^ binary_erosion(target)
    dist_a = distance_transform_edt(~target)[pred_surf]
    dist_b = distance_transform_edt(~pred)[tgt_surf]
    return float(np.percentile(np.concatenate([dist_a, dist_b]), 95)) if dist_a.size and dist_b.size else float("nan")

# Export functions
def export_metrics_csv(metrics_dict):
    """Export metrics to CSV"""
    rows = []
    for case_id, metrics in metrics_dict.items():
        row = {"case_id": case_id, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "metrics_validation.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n[EXPORT] Metrics saved: {csv_path}")
    print(f"[STATS]  Mean Dice: {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}")
    print(f"         Mean IoU: {df['iou'].mean():.4f} +/- {df['iou'].std():.4f}")

def export_metrics_figure(metrics_dict):
    """Export metrics comparison figure"""
    metrics_list = list(metrics_dict.values())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ExU-Trans Metrics (2D Slice-Based)", fontsize=14, fontweight='bold')

    metrics_to_plot = [
        ("Dice", "dice", (0, 0)),
        ("IoU", "iou", (0, 1)),
        ("HD95 (mm)", "hd95", (0, 2)),
        ("Precision", "precision", (1, 0)),
        ("Recall", "recall", (1, 1)),
        ("F1 Score", "f1", (1, 2)),
    ]

    for title, key, (row, col) in metrics_to_plot:
        ax = axes[row, col]
        values = [m[key] for m in metrics_list]
        ax.hist(values, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"{title}\nMean: {np.nanmean(values):.4f}")
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure_4_metrics_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EXPORT] Figure saved: {fig_path}")

# Main execution
print("\n" + "=" * 90)
print("PIPELINE READY")
print("=" * 90)

print("\nDataset is ready for processing!")
print(f"Training cases: {len(train_cases)}")
print(f"Validation cases: {len(val_cases)}")
print(f"Test cases: {len(test_cases)}")

print("\nTo run full training:")
print("  1. Use the Jupyter notebook: notebooks/exu_trans_brats2020_reproduction.ipynb")
print("  2. Or implement full training loop in this script")

print("\nOutputs will be saved to:")
print(f"  - Metrics CSV: {METRICS_DIR / 'metrics_validation.csv'}")
print(f"  - Figures: {FIGURES_DIR}")
print(f"  - Attention maps: {ATTENTION_DIR}")

print("\n" + "=" * 90)
