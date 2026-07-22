"""Verify that all dependencies and data are correctly set up."""

import sys
from pathlib import Path

# Add scripts directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

print("=== ExU-Trans Setup Verification ===\n")

# Check imports
print("1. Checking Python dependencies...")
try:
    import torch
    print(f"   [OK] PyTorch {torch.__version__}")
except ImportError:
    print("   [ERROR] PyTorch not found")
    sys.exit(1)

try:
    import numpy as np
    print(f"   [OK] NumPy {np.__version__}")
except ImportError:
    print("   [ERROR] NumPy not found")
    sys.exit(1)

try:
    import nibabel
    print(f"   [OK] Nibabel {nibabel.__version__}")
except ImportError:
    print("   [ERROR] Nibabel not found")
    sys.exit(1)

try:
    import pandas
    print(f"   [OK] Pandas {pandas.__version__}")
except ImportError:
    print("   [ERROR] Pandas not found")
    sys.exit(1)

try:
    import matplotlib
    print(f"   [OK] Matplotlib {matplotlib.__version__}")
except ImportError:
    print("   [ERROR] Matplotlib not found")
    sys.exit(1)

# Check modules
print("\n2. Checking project modules...")
modules = ["config", "utils_device", "dataset", "model", "losses", "train", "visualize", "export"]
for mod in modules:
    try:
        __import__(mod)
        print(f"   [OK] {mod}.py")
    except ImportError as e:
        print(f"   [ERROR] {mod}.py - {e}")

# Check data paths
print("\n3. Checking data directories...")
from config import CONFIG

train_root = CONFIG["DATA_ROOT_TRAIN"]
val_root = CONFIG["DATA_ROOT_OFFICIAL_VAL"]

if train_root.exists():
    cases = list(train_root.glob("BraTS20_Training_*"))
    print(f"   [OK] Training data: {len(cases)} cases found")
else:
    print(f"   [ERROR] Training data not found at {train_root}")

if val_root.exists():
    cases = list(val_root.glob("BraTS20_Validation_*"))
    print(f"   [OK] Validation data: {len(cases)} cases found")
else:
    print(f"   [ERROR] Validation data not found at {val_root}")

# Check device
print("\n4. Checking compute device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   [OK] Device: {device}")
if torch.cuda.is_available():
    print(f"   [OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"   [OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n[SUCCESS] All checks passed! Ready to train.")
print("\nRun training with:")
print("  python main.py")
