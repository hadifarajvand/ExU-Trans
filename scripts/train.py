"""Training and case-level evaluation loops."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch
from tqdm.auto import tqdm

from losses import (
    segmentation_loss, to_pred, dice_score, iou_score,
    precision_score_np, recall_score_np, f1_score_np, hd95,
)


def train_one_epoch(
    model, loader, optimizer, label_mode, device,
    pixel_weight=1.0, align_weight=0.05, boundary_weight=0.05,
):
    model.train()
    total = 0.0
    sums = defaultdict(float)
    for images, masks, _, _ in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(images)
        loss, parts = segmentation_loss(
            logits, masks, aux, label_mode,
            pixel_weight, align_weight, boundary_weight,
        )
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        for k, v in parts.items():
            sums[k] += float(v.item())
    n = max(len(loader), 1)
    return total / n, {k: v / n for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, label_mode, device):
    """Reconstruct each evaluated case from 2D predictions, then score per case."""
    model.eval()
    by_case = defaultdict(dict)
    sample_preview = []
    for images, masks, case_id, slice_idx in tqdm(loader, desc="eval", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits, aux = model(images)
        pred = to_pred(logits, label_mode).cpu().numpy().squeeze()
        target = masks.cpu().numpy().squeeze()
        if label_mode != "whole_tumor":
            pred = pred > 0
            target = target > 0
        name = case_id[0] if isinstance(case_id, (tuple, list)) else str(case_id)
        idx = int(slice_idx[0].item() if isinstance(slice_idx, torch.Tensor) else slice_idx)
        by_case[name][idx] = (pred.astype(np.uint8), target.astype(np.uint8))
        if len(sample_preview) < 8:
            sample_preview.append((images.cpu(), masks.cpu(), pred, name, idx, aux))

    all_metrics = {}
    for name, slices in by_case.items():
        ordered = [slices[i] for i in sorted(slices)]
        pred_volume = np.stack([x[0] for x in ordered], axis=0)
        target_volume = np.stack([x[1] for x in ordered], axis=0)
        all_metrics[name] = {
            "dice": dice_score(pred_volume, target_volume),
            "iou": iou_score(pred_volume, target_volume),
            "precision": precision_score_np(pred_volume, target_volume),
            "recall": recall_score_np(pred_volume, target_volume),
            "f1": f1_score_np(pred_volume, target_volume),
            # Resizing and missing calibrated physical spacing make this px/grid, not mm.
            "hd95_px": hd95(pred_volume, target_volume),
        }

    keys = ["dice", "iou", "precision", "recall", "f1", "hd95_px"]
    summary = {
        k: float(np.nanmean([m[k] for m in all_metrics.values()])) if all_metrics else float("nan")
        for k in keys
    }
    return summary, sample_preview, all_metrics
