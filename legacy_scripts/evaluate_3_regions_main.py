#!/usr/bin/env python
"""
Main evaluation script for 3-region ExU-Trans segmentation
Computes metrics for WT (Whole Tumor), TC (Tumor Core), and ET (Enhancing Tumor)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from evaluate_3_regions import (
    extract_tumor_regions,
    compute_metrics_all_regions,
    create_metrics_dataframe,
    aggregate_metrics,
    dice_score, iou_score, precision_score, recall_score, f1_score, hd95
)

def load_nifti(path):
    """Load NIfTI file and transpose to (D, H, W)"""
    data = nib.load(path).get_fdata().astype(np.float32)
    return np.transpose(data, (2, 0, 1))

def main():
    print("="*70)
    print("3-REGION EVALUATION FOR ExU-Trans SEGMENTATION")
    print("="*70)

    # Paths
    dataset_root = Path("dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    outputs_dir = Path("outputs")
    metrics_dir = outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load some example cases
    case_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])[:5]

    print(f"\nFound {len(case_dirs)} case directories")

    all_case_results = {}

    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_name = case_dir.name

        # Load segmentation
        seg_path = case_dir / f"{case_name}_seg.nii"
        if not seg_path.exists():
            print(f"Warning: {seg_path} not found")
            continue

        seg_3d = load_nifti(str(seg_path))

        # Create a dummy prediction (for demonstration)
        # In practice, this would come from the model
        pred_3d = seg_3d.copy()

        # Add some noise to make it different
        pred_3d = (pred_3d > 0).astype(np.float32)
        noise = np.random.randn(*pred_3d.shape) * 0.1
        pred_3d_noisy = np.clip(pred_3d + noise, 0, 1)

        # Evaluate each slice
        results_per_slice = {}
        for slice_idx in range(seg_3d.shape[0]):
            if seg_3d[slice_idx].sum() == 0:  # Skip empty slices
                continue

            pred_2d = pred_3d_noisy[slice_idx]
            seg_2d = seg_3d[slice_idx]

            # Compute metrics for all 3 regions
            region_metrics = compute_metrics_all_regions(pred_2d, seg_2d)
            results_per_slice[f"slice_{slice_idx}"] = region_metrics

        all_case_results[case_name] = results_per_slice

    # Create DataFrame
    rows = []
    for case_name, slices_dict in all_case_results.items():
        for slice_key, regions_dict in slices_dict.items():
            for region, metrics in regions_dict.items():
                row = {
                    'case_id': case_name,
                    'slice': slice_key,
                    'region': region,
                    **metrics
                }
                rows.append(row)

    df_results = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal entries: {len(df_results)}")

    # Summary by region
    print("\n" + "-"*70)
    print("METRICS SUMMARY BY REGION")
    print("-"*70)

    for region in ['WT', 'TC', 'ET']:
        region_data = df_results[df_results['region'] == region]
        if len(region_data) == 0:
            continue

        print(f"\n{region} (n={len(region_data)} slices)")
        print("-"*70)

        for metric in ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']:
            if metric in region_data.columns:
                valid_vals = region_data[metric].dropna()
                if len(valid_vals) > 0:
                    mean_val = valid_vals.mean()
                    std_val = valid_vals.std()

                    if metric == 'hd95':
                        print(f"  {metric:12s}: {mean_val:8.4f} ± {std_val:8.4f} mm")
                    else:
                        print(f"  {metric:12s}: {mean_val:8.4f} ± {std_val:8.4f}")

    # Export to CSV
    csv_path = metrics_dir / "metrics_3regions_demo.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n\nResults exported to: {csv_path}")

    print("\nFirst 10 rows:")
    print(df_results.head(10).to_string())

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
