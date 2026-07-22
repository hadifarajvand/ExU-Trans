"""
Functions to evaluate ExU-Trans on 3 tumor regions: WT, TC, ET
This module computes metrics separately for each region as in the paper.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from functools import lru_cache
import nibabel as nib


def extract_tumor_regions(seg_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract WT, TC, ET regions from segmentation mask

    Seg values: 0=background, 1=NCR (necrotic), 2=ED (edema), 4=ET (enhancing)
    WT = 1+2+4 (Whole Tumor)
    TC = 1+4   (Tumor Core)
    ET = 4     (Enhancing Tumor)
    """
    wt = ((seg_mask == 1) | (seg_mask == 2) | (seg_mask == 4)).astype(np.float32)
    tc = ((seg_mask == 1) | (seg_mask == 4)).astype(np.float32)
    et = (seg_mask == 4).astype(np.float32)
    return {'WT': wt, 'TC': tc, 'ET': et}


def dice_score(pred, target, eps=1e-6):
    """Dice Similarity Coefficient"""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred, target, eps=1e-6):
    """Intersection over Union (Jaccard Index)"""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))


def precision_score(pred, target, eps=1e-6):
    """Precision (True Positive Rate)"""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score(pred, target, eps=1e-6):
    """Recall (Sensitivity)"""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    return float((tp + eps) / (tp + fn + eps))


def f1_score(pred, target, eps=1e-6):
    """F1 Score (harmonic mean of precision and recall)"""
    p = precision_score(pred, target, eps)
    r = recall_score(pred, target, eps)
    return float((2 * p * r + eps) / (p + r + eps))


def hd95(pred, target):
    """Hausdorff Distance (95th percentile)"""
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    pred_surf = pred ^ binary_erosion(pred)
    tgt_surf = target ^ binary_erosion(target)
    dist_a = distance_transform_edt(~target)[pred_surf]
    dist_b = distance_transform_edt(~pred)[tgt_surf]
    if dist_a.size and dist_b.size:
        return float(np.percentile(np.concatenate([dist_a, dist_b]), 95))
    else:
        return float("nan")


def compute_metrics_all_regions(pred: np.ndarray, seg_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute metrics for all 3 regions (WT, TC, ET)

    Args:
        pred: Model prediction (binary or continuous)
        seg_mask: Full segmentation mask with region labels

    Returns:
        Dict mapping region names to their metrics dict
    """
    # Ensure pred is binary
    pred_binary = (pred > 0.5).astype(np.float32) if pred.dtype == np.float32 else pred.astype(np.float32)

    # Extract regions
    regions = extract_tumor_regions(seg_mask)

    # Compute metrics for each region
    results = {}
    for region_name, region_mask in regions.items():
        # Skip if region doesn't exist in ground truth
        if region_mask.sum() == 0:
            results[region_name] = {
                'dice': 0.0,
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'hd95': float('nan')
            }
        else:
            results[region_name] = {
                'dice': dice_score(pred_binary, region_mask),
                'iou': iou_score(pred_binary, region_mask),
                'precision': precision_score(pred_binary, region_mask),
                'recall': recall_score(pred_binary, region_mask),
                'f1': f1_score(pred_binary, region_mask),
                'hd95': hd95(pred_binary, region_mask)
            }

    return results


def aggregate_metrics(all_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across all cases

    Args:
        all_results: List of result dicts from compute_metrics_all_regions

    Returns:
        Dict with mean and std for each region and metric
    """
    aggregated = {}

    # Get all regions from first result
    if not all_results:
        return {}

    regions = all_results[0].keys()
    metrics = all_results[0][list(regions)[0]].keys()

    for region in regions:
        aggregated[region] = {}
        for metric in metrics:
            values = [r[region][metric] for r in all_results]
            # Filter out NaN for mean/std calculation
            valid_values = [v for v in values if not np.isnan(v)]

            if valid_values:
                aggregated[region][f'{metric}_mean'] = float(np.mean(valid_values))
                aggregated[region][f'{metric}_std'] = float(np.std(valid_values))
            else:
                aggregated[region][f'{metric}_mean'] = float('nan')
                aggregated[region][f'{metric}_std'] = float('nan')

    return aggregated


def create_metrics_dataframe(all_case_results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """Create a pandas DataFrame with metrics for all cases and regions

    Args:
        all_case_results: Dict mapping case_id -> region -> metrics

    Returns:
        DataFrame with columns: case_id, region, dice, iou, precision, recall, f1, hd95
    """
    rows = []

    for case_id, regions_dict in all_case_results.items():
        for region, metrics in regions_dict.items():
            row = {'case_id': case_id, 'region': region, **metrics}
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Region evaluation functions loaded successfully")
    print("\nAvailable functions:")
    print("  - extract_tumor_regions(seg_mask)")
    print("  - compute_metrics_all_regions(pred, seg_mask)")
    print("  - aggregate_metrics(all_results)")
    print("  - create_metrics_dataframe(all_case_results)")
