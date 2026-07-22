"""Device setup and seeding utilities."""

import random
import numpy as np
import torch
from config import CONFIG, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    """Setup device and print configuration."""
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("device:", device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n=== CONFIGURATION ===")
    for key, value in sorted(CONFIG.items()):
        print(f"  {key}: {value}")
    print("\nDataset split explanation:")
    print("  - Labeled training cases (n=369) are split internally: 70% train, 15% val, 15% test")
    print("  - Official validation cases (n=125) are treated as unlabeled inference-only")
    print(f"  - Mode: {CONFIG['PATCH_OR_SLICE_MODE'].upper()} (2D slice-based, not 3D)")

    return device
