from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from losses import segmentation_loss, to_pred, dice_score, hd95, iou_score, precision_score_np, recall_score_np, f1_score_np


def train_one_epoch(model, loader, optimizer, label_mode, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        outputs, aux = model(images)
        loss, _ = segmentation_loss(outputs, masks, aux, label_mode=label_mode)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, label_mode, device):
    model.eval()
    by_case = {}
    sample_preview = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            case_id = batch["volume_id"]
            slice_idx = batch["slice_id"]
            outputs, _ = model(images)
            pred = to_pred(outputs, label_mode=label_mode).detach().cpu().numpy()
            target = masks.detach().cpu().numpy()
            if target.ndim == 4:
                target = target[:, 0]
            if label_mode == "whole_tumor":
                pred = pred > 0
                target = target > 0
            name = str(case_id[0] if isinstance(case_id, (tuple, list)) else case_id)
            idx = int(slice_idx[0].item() if isinstance(slice_idx, torch.Tensor) else slice_idx[0])
            by_case.setdefault(name, {})[idx] = (pred.astype(np.uint8), target.astype(np.uint8))
            if len(sample_preview) < 8:
                sample_preview.append((images.cpu(), masks.cpu(), pred, name, idx, batch.get("metadata", {})))

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
            "hd95_px": hd95(pred_volume, target_volume),
        }
    keys = ["dice", "iou", "precision", "recall", "f1", "hd95_px"]
    summary = {k: float(np.nanmean([m[k] for m in all_metrics.values()])) if all_metrics else float('nan') for k in keys}
    return summary, sample_preview, all_metrics
