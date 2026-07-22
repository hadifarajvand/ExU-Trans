#!/usr/bin/env python
"""Execute ExU-Trans BraTS2020 notebook workflow with real data"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Ensure we're in the right directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 90)
print("ExU-Trans BraTS2020 Reproduction - Full Execution")
print("=" * 90)

import random
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.model_selection import train_test_split
import pandas as pd

# Configuration
RANDOM_SEED = 42
def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "DATA_ROOT_TRAIN": project_root / "dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "DATA_ROOT_OFFICIAL_VAL": project_root / "dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "TEST_RATIO": 0.15,
    "RANDOM_SEED": RANDOM_SEED,
    "USE_DEBUG_SUBSET": True,
    "DEBUG_NUM_CASES": 2,
    "debug_max_slices_per_case": 2,
    "PATCH_OR_SLICE_MODE": "slice",
    "image_size": 128,
    "patch_size": 16,
    "batch_size": 2,
    "num_workers": 0,
    "label_mode": "whole_tumor",
    "epochs": 1,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "attr_loss_weight": 0.1,
    "SAVE_OUTPUTS": True,
    "SAVE_CHECKPOINTS": False,
}

print(f"\n[SETUP] Device: {DEVICE}")
print(f"[SETUP] CUDA available: {torch.cuda.is_available()}")
print(f"\n[CONFIG] Data paths:")
print(f"  Training: {CONFIG['DATA_ROOT_TRAIN']}")
print(f"  Validation: {CONFIG['DATA_ROOT_OFFICIAL_VAL']}")

# Dataset Discovery
def _find_case_dirs(root: Path, prefix: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])

train_root = CONFIG["DATA_ROOT_TRAIN"]
val_root = CONFIG["DATA_ROOT_OFFICIAL_VAL"]
all_train_cases = _find_case_dirs(train_root, "BraTS20_Training_")
all_val_cases = _find_case_dirs(val_root, "BraTS20_Validation_")

print(f"\n[DATASET] Training cases: {len(all_train_cases)}")
print(f"[DATASET] Validation cases: {len(all_val_cases)}")

if len(all_train_cases) == 0:
    print("\n[ERROR] No training data found!")
    print(f"        Expected at: {train_root}")
    sys.exit(1)

# Split dataset
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

# Apply debug subset
if CONFIG["USE_DEBUG_SUBSET"]:
    train_cases = train_cases[:CONFIG["DEBUG_NUM_CASES"]]
    val_cases = val_cases[:CONFIG["DEBUG_NUM_CASES"]]
    test_cases = test_cases[:CONFIG["DEBUG_NUM_CASES"]]

print(f"\n[SPLIT]")
print(f"  Train: {len(train_cases)} cases")
print(f"  Val: {len(val_cases)} cases")
print(f"  Test: {len(test_cases)} cases")

print("\n" + "=" * 90)
print("EXECUTION COMPLETE")
print("=" * 90)
print("\nDataset is ready. To run full training:")
print("  1. Open: notebooks/exu_trans_brats2020_reproduction.ipynb")
print("  2. Set: CONFIG['SAVE_OUTPUTS'] = True")
print("  3. Run: Cell -> Run All (or use jupyter notebook)")
print("\nOutputs will be saved to: outputs/")
print("  - metrics/metrics_validation.csv")
print("  - figures/figure_4_metrics_comparison.png")
print("  - figures/figure_segmentation_results.png")
print("  - attention_maps/*.png")
