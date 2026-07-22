"""Portable configuration for the ExU-Trans approximation."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = int(os.getenv("EXUTRANS_SEED", "42"))
DATASET_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "dataset"))

CONFIG = {
    "DATA_ROOT_TRAIN": Path(os.getenv(
        "DATA_ROOT_TRAIN",
        DATASET_ROOT / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
    )),
    "DATA_ROOT_OFFICIAL_VAL": Path(os.getenv(
        "DATA_ROOT_OFFICIAL_VAL",
        DATASET_ROOT / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData",
    )),
    # The paper does not publish exact split IDs. These ratios are a repository
    # assumption and are recorded as such in every measured run manifest.
    "TRAIN_RATIO": float(os.getenv("TRAIN_RATIO", "0.70")),
    "VAL_RATIO": float(os.getenv("VAL_RATIO", "0.15")),
    "TEST_RATIO": float(os.getenv("TEST_RATIO", "0.15")),
    "RANDOM_SEED": RANDOM_SEED,
    "USE_DEBUG_SUBSET": os.getenv("USE_DEBUG_SUBSET", "0") == "1",
    "DEBUG_NUM_CASES": int(os.getenv("DEBUG_NUM_CASES", "2")),
    "debug_max_slices_per_case": int(os.getenv("DEBUG_MAX_SLICES_PER_CASE", "2")),
    "PATCH_OR_SLICE_MODE": "slice",
    # The paper says scans are resized but does not publish the exact target size.
    "image_size": int(os.getenv("IMAGE_SIZE", "128")),
    "patch_size": int(os.getenv("PATCH_SIZE", "16")),
    "batch_size": int(os.getenv("BATCH_SIZE", "4")),
    "num_workers": int(os.getenv("NUM_WORKERS", "0")),
    "label_mode": os.getenv("LABEL_MODE", "whole_tumor"),
    "epochs": int(os.getenv("EPOCHS", "100")),
    # Paper Table 13 identifies 0.0005 and 4 transformer layers as optimal.
    "lr": float(os.getenv("LEARNING_RATE", "0.0005")),
    "num_transformer_layers": int(os.getenv("NUM_TRANSFORMER_LAYERS", "4")),
    "weight_decay": float(os.getenv("WEIGHT_DECAY", "0.0001")),
    "pixel_loss_weight": float(os.getenv("PIXEL_LOSS_WEIGHT", "1.0")),
    "align_loss_weight": float(os.getenv("ALIGN_LOSS_WEIGHT", "0.05")),
    "boundary_loss_weight": float(os.getenv("BOUNDARY_LOSS_WEIGHT", "0.05")),
    "SAVE_OUTPUTS": True,
    "SAVE_CHECKPOINTS": True,
}

assert abs(CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"] + CONFIG["TEST_RATIO"] - 1.0) < 1e-6

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MEASURED_DIR = OUTPUT_DIR / "measured"
METRICS_DIR = MEASURED_DIR / "metrics"
FIGURES_DIR = MEASURED_DIR / "figures"
PREDICTIONS_DIR = MEASURED_DIR / "predictions"
ATTENTION_DIR = MEASURED_DIR / "attention_maps"
CHECKPOINT_DIR = MEASURED_DIR / "checkpoints"
