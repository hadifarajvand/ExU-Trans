#!/usr/bin/env python
"""Execute the ExU-Trans notebook with error handling"""
import subprocess
import sys
from pathlib import Path

notebook_path = Path("notebooks/exu_trans_brats2020_reproduction.ipynb")

if not notebook_path.exists():
    print(f"Error: {notebook_path} not found")
    sys.exit(1)

print("=" * 80)
print(f"Executing: {notebook_path}")
print("=" * 80)

# Use jupyter nbconvert to execute
result = subprocess.run([
    sys.executable, "-m", "nbconvert",
    "--to", "notebook",
    "--inplace",
    "--execute",
    "--ExecutePreprocessor.timeout=600",
    "--allow-errors",
    str(notebook_path)
], capture_output=False, text=True)

sys.exit(result.returncode)
