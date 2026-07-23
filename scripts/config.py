from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "dataset"))
DEFAULT_DATASET_DIR = DATASET_ROOT / "fulldataset" / "BraTS2020_training_data" / "content" / "data"
DEFAULT_METADATA_CSV = DATASET_ROOT / "fulldataset" / "BraTS20 Training Metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MEASURED_DIR = OUTPUT_ROOT / "measured"
PREVIEW_DIR = OUTPUT_ROOT / "preflight"
SMOKE_DIR = OUTPUT_ROOT / "smoke"

CONFIG = {
    "PROJECT_ROOT": PROJECT_ROOT,
    "DATASET_ROOT": DATASET_ROOT,
    "DATASET_DIR": DEFAULT_DATASET_DIR,
    "metadata_csv": DEFAULT_METADATA_CSV,
    "split_dir": MEASURED_DIR / "splits",
    "OUTPUT_ROOT": OUTPUT_ROOT,
    "MEASURED_DIR": MEASURED_DIR,
    "METRICS_DIR": MEASURED_DIR / "metrics",
    "FIGURES_DIR": MEASURED_DIR / "figures",
    "PREDICTIONS_DIR": MEASURED_DIR / "predictions",
    "ATTENTION_DIR": MEASURED_DIR / "attention",
    "CHECKPOINT_DIR": MEASURED_DIR / "checkpoints",
    "PREVIEW_DIR": PREVIEW_DIR,
    "SMOKE_DIR": SMOKE_DIR,
    "device_preference": ["cuda", "mps", "cpu"],
    "batch_size": int(os.getenv("BATCH_SIZE", "8")),
    "num_workers": int(os.getenv("NUM_WORKERS", "0")),
    "image_size": int(os.getenv("IMAGE_SIZE", "240")),
    "label_mode": os.getenv("LABEL_MODE", "regions"),
    "TRAIN_RATIO": 0.7,
    "VAL_RATIO": 0.15,
    "TEST_RATIO": 0.15,
    "RANDOM_SEED": int(os.getenv("SEED", "42")),
    "USE_DEBUG_SUBSET": os.getenv("USE_DEBUG_SUBSET", "0") == "1",
    "DEBUG_NUM_CASES": int(os.getenv("DEBUG_NUM_CASES", "2")),
    "debug_max_slices_per_case": int(os.getenv("DEBUG_MAX_SLICES_PER_CASE", "4")),
    "learning_rate": float(os.getenv("LEARNING_RATE", "0.0005")),
    "transformer_layers": int(os.getenv("TRANSFORMER_LAYERS", "4")),
}

RANDOM_SEED = CONFIG["RANDOM_SEED"]
METRICS_DIR = CONFIG["METRICS_DIR"]
FIGURES_DIR = CONFIG["FIGURES_DIR"]
PREDICTIONS_DIR = CONFIG["PREDICTIONS_DIR"]
ATTENTION_DIR = CONFIG["ATTENTION_DIR"]
CHECKPOINT_DIR = CONFIG["CHECKPOINT_DIR"]
