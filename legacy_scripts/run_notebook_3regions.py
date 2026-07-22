#!/usr/bin/env python
"""
Run the 3-region evaluation on a trained ExU-Trans model
This is a standalone script that uses the notebook's model and functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import everything from the module
from evaluate_3_regions import (
    extract_tumor_regions,
    compute_metrics_all_regions,
    create_metrics_dataframe,
    aggregate_metrics
)

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

print("="*60)
print("3-Region Evaluation Script")
print("="*60)

if __name__ == "__main__":
    print("\nRunning 3-region evaluation...")
