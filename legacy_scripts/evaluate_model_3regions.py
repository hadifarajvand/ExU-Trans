#!/usr/bin/env python
"""
3-Region evaluation using actual model predictions
Loads the trained model and evaluates on validation set for WT, TC, ET
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache
from typing import Dict, Optional, Tuple

from evaluate_3_regions import (
    extract_tumor_regions,
    compute_metrics_all_regions,
    create_metrics_dataframe,
)

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
DATASET_ROOT = Path("dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
OUTPUTS_DIR = Path("outputs")
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


@lru_cache(maxsize=16)
def load_nifti(path: str) -> np.ndarray:
    """Load NIfTI file and transpose to (D, H, W)"""
    data = nib.load(path).get_fdata().astype(np.float32)
    return np.transpose(data, (2, 0, 1))


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization per non-background voxels"""
    mask = volume > 0
    if mask.any():
        mean = float(volume[mask].mean())
        std = float(volume[mask].std())
        std = 1.0 if std < 1e-6 else std
        volume = (volume - mean) / std
    return volume.astype(np.float32)


def load_case(case_dir: Path) -> Tuple[Dict, Optional[np.ndarray]]:
    """Load all modalities and segmentation for a case"""
    files = {}
    for path in case_dir.glob("*.nii*"):
        n = path.name.lower()
        if '_flair' in n:
            files['flair'] = path
        elif n.endswith('_t1.nii') or n.endswith('_t1.nii.gz'):
            files['t1'] = path
        elif '_t1ce' in n or '_t1gd' in n:
            files['t1ce'] = path
        elif '_t2' in n:
            files['t2'] = path
        elif '_seg' in n:
            files['seg'] = path

    modalities = {}
    for key in ['flair', 't1', 't1ce', 't2']:
        if key in files:
            modalities[key] = zscore_normalize(load_nifti(str(files[key])))

    seg = None
    if 'seg' in files:
        seg = load_nifti(str(files['seg'])).astype(np.int64)

    return modalities, seg


# ============================================================================
# Model Architecture (ExUTransLite)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (h, w)


class SEMHABlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.eps = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.n2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x):
        x1 = self.n1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + attn_out + 0.1 * self.eps(x1)
        x = x + self.mlp(self.n2(x))
        return x


class DAE(nn.Module):
    def __init__(self, embed_dim, attr_dim=4):
        super().__init__()
        self.token_score = nn.Linear(embed_dim, 1)
        self.attr_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, attr_dim))

    def forward(self, tokens, hw):
        attr_logits = self.attr_head(tokens.mean(dim=1))
        attr_map = torch.sigmoid(self.token_score(tokens)).transpose(1, 2).reshape(tokens.shape[0], 1, hw[0], hw[1])
        return attr_logits, attr_map


class ContextualSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        hidden = max(channels // 4, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg, mx = x.mean(1, keepdim=True), x.amax(1, keepdim=True)
        spatial = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        ch = self.channel(x)
        return x * spatial * ch, spatial, ch


class BivariateFusion(nn.Module):
    def __init__(self, local_ch, global_ch):
        super().__init__()
        self.local = nn.Conv2d(local_ch, local_ch, 1)
        self.global_ = nn.Conv2d(global_ch, local_ch, 1)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        hidden = max(local_ch // 4, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(local_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, local_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        local_feat = self.local(local_feat)
        global_feat = self.global_(global_feat)
        fused = local_feat + global_feat
        spatial = torch.sigmoid(self.spatial(torch.cat([fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)], dim=1)))
        ch = self.channel(fused)
        return fused * spatial * ch + local_feat, spatial, ch


class ExUTransLite(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, patch_size=16, embed_dim=128):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)
        self.patch = PatchEmbed(in_ch, embed_dim, patch_size)
        self.sem1 = SEMHABlock(embed_dim)
        self.sem2 = SEMHABlock(embed_dim)
        self.dae = DAE(embed_dim)
        self.glob = nn.Conv2d(embed_dim, 256, 1)
        self.fuse = BivariateFusion(256, 256)
        self.ctx = ContextualSelfAttention(256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.enc4(s3)

        tokens, hw = self.patch(x)
        tokens = self.sem2(self.sem1(tokens))
        attr_logits, attr_map = self.dae(tokens, hw)

        global_map = tokens.transpose(1, 2).reshape(x.shape[0], -1, hw[0], hw[1])
        global_map = F.interpolate(global_map, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        global_map = self.glob(global_map)

        fused, fusion_spatial, fusion_channel = self.fuse(bottleneck, global_map)
        fused, ctx_spatial, ctx_channel = self.ctx(fused)

        d3 = self.up3(fused, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        out = self.head(d1)

        return out, {"attr_logits": attr_logits, "attr_map": F.interpolate(attr_map, size=x.shape[-2:], mode="bilinear", align_corners=False)}


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("="*80)
    print("3-REGION EVALUATION WITH MODEL PREDICTIONS")
    print("="*80)

    # Load model
    print("\nInitializing model...")
    model = ExUTransLite(in_ch=4, out_ch=1, patch_size=16, embed_dim=128).to(DEVICE)
    model.eval()

    # Get validation cases
    case_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])[55:60]  # Use validation subset
    print(f"Processing {len(case_dirs)} cases...")

    all_case_results = {}

    for case_dir in tqdm(case_dirs, desc="Evaluating"):
        case_name = case_dir.name

        try:
            modalities, seg_3d = load_case(case_dir)
            if seg_3d is None or len(modalities) < 4:
                continue

            # Stack modalities
            mods_stack = np.stack([modalities['flair'], modalities['t1'], modalities['t1ce'], modalities['t2']], axis=0)

            # Process each slice
            for slice_idx in range(min(seg_3d.shape[0], 50)):  # Limit slices
                if seg_3d[slice_idx].sum() == 0:
                    continue

                seg_2d = seg_3d[slice_idx]
                img_2d = mods_stack[:, slice_idx, :, :]

                # Resize to 128x128
                img_2d = F.interpolate(
                    torch.from_numpy(img_2d).unsqueeze(0).float(),
                    size=(128, 128),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                seg_2d_resized = F.interpolate(
                    torch.from_numpy(seg_2d).unsqueeze(0).unsqueeze(0).float(),
                    size=(128, 128),
                    mode='nearest'
                ).squeeze().numpy()

                # Model prediction
                with torch.no_grad():
                    logits, _ = model(img_2d.unsqueeze(0).to(DEVICE))
                    pred = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Resize back
                pred_orig = F.interpolate(
                    torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
                    size=seg_2d.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()

                # Compute metrics for 3 regions
                region_metrics = compute_metrics_all_regions(pred_orig, seg_2d)

                key = f"{case_name}_slice_{slice_idx}"
                all_case_results[key] = region_metrics

        except Exception as e:
            continue

    if len(all_case_results) == 0:
        print("No results collected!")
        return

    # Create DataFrame
    df_3regions = create_metrics_dataframe(all_case_results)

    print(f"\nEvaluated {len(all_case_results)} slices")
    print(f"Total {len(df_3regions)} region entries\n")

    # Print summary
    print("="*80)
    print("RESULTS BY REGION")
    print("="*80)

    for region in ['WT', 'TC', 'ET']:
        reg_data = df_3regions[df_3regions['region'] == region]
        if len(reg_data) == 0:
            continue

        print(f"\n{region} Region (n={len(reg_data)} slices)")
        print("-" * 80)
        for metric in ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']:
            valid = reg_data[metric].dropna()
            if len(valid) > 0:
                m = valid.mean()
                s = valid.std()
                if metric == 'hd95':
                    print(f"  {metric:12s}: {m:8.4f} ± {s:8.4f} mm")
                else:
                    print(f"  {metric:12s}: {m:8.4f} ± {s:8.4f}")

    # Export results
    csv_path = METRICS_DIR / "metrics_3regions_model.csv"
    df_3regions.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
