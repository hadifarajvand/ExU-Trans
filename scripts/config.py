"""Configuration for ExU-Trans BraTS2020 reproduction."""

from pathlib import Path

RANDOM_SEED = 42

CONFIG = {
    "DATA_ROOT_TRAIN": Path(r"D:\GitHub\aysan\class-projects\1\dataset\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"),
    "DATA_ROOT_OFFICIAL_VAL": Path(r"D:\GitHub\aysan\class-projects\1\dataset\BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData"),
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "TEST_RATIO": 0.15,
    "RANDOM_SEED": RANDOM_SEED,
    "USE_DEBUG_SUBSET": False,
    "DEBUG_NUM_CASES": 2,
    "debug_max_slices_per_case": 2,
    "PATCH_OR_SLICE_MODE": "slice",
    "image_size": 128,
    "patch_size": 16,
    "batch_size": 4,
    "num_workers": 0,
    "label_mode": "whole_tumor",
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "attr_loss_weight": 0.1,
    "SAVE_OUTPUTS": True,
    "SAVE_CHECKPOINTS": False,
}

assert abs((CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"] + CONFIG["TEST_RATIO"]) - 1.0) < 1e-6

# Output directories
OUTPUT_DIR = Path("outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
ATTENTION_DIR = OUTPUT_DIR / "attention_maps"
