"""Device setup and deterministic seeding utilities."""
import random
import numpy as np
import torch

from config import CONFIG, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n=== CONFIGURATION ===")
    for key, value in sorted(CONFIG.items()):
        print(f"  {key}: {value}")
    print("\nReproducibility note:")
    print("  - The 70/15/15 internal split is a repository assumption, not a published split-ID list.")
    print("  - Measured outputs are kept separate from paper reference values.")
    print("  - HD95 is not called millimetres unless physical spacing is explicitly calibrated.")
    return device
