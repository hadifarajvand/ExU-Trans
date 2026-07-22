"""Training and evaluation loops."""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from losses import segmentation_loss, to_pred, dice_score, iou_score, precision_score_np, recall_score_np, f1_score_np, hd95


def train_one_epoch(model, loader, optimizer, label_mode, device, attr_weight=0.1, align_weight=0.05, boundary_weight=0.05):
    """Train for one epoch."""
    model.train()
    total = 0.0
    loss_components = {"bce": 0, "dice": 0, "attr": 0, "align": 0, "boundary": 0}

    for images, masks, _, _ in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(images)
        loss, loss_dict = segmentation_loss(logits, masks, aux, label_mode, attr_weight, align_weight, boundary_weight)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += float(loss_dict[k].item())

    return total / max(len(loader), 1), {k: v / max(len(loader), 1) for k, v in loss_components.items()}


@torch.no_grad()
def evaluate(model, loader, label_mode, device):
    """Evaluate on dataset."""
    model.eval()
    scores = {"dice": [], "iou": [], "precision": [], "recall": [], "f1": [], "hd95": []}
    samples = []
    all_metrics = {}

    for images, masks, case_id, slice_idx in tqdm(loader, desc="eval", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits, aux = model(images)

        pred = to_pred(logits, label_mode).cpu().numpy().squeeze()
        target = masks.cpu().numpy().squeeze()
        if label_mode != "whole_tumor":
            target = (target > 0).astype(np.int64)

        dice = dice_score(pred, target)
        iou = iou_score(pred, target)
        precision = precision_score_np(pred, target)
        recall = recall_score_np(pred, target)
        f1 = f1_score_np(pred, target)
        hd = hd95(pred, target)

        scores["dice"].append(dice)
        scores["iou"].append(iou)
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["f1"].append(f1)
        scores["hd95"].append(hd)

        case_name = case_id[0] if isinstance(case_id, tuple) else case_id
        case_key = f"{case_name}_slice_{slice_idx[0].item() if isinstance(slice_idx, torch.Tensor) else slice_idx}"
        all_metrics[case_key] = {
            "dice": dice, "iou": iou, "precision": precision,
            "recall": recall, "f1": f1, "hd95": hd
        }
        samples.append((images.cpu(), masks.cpu(), pred, case_id, slice_idx, aux))

    return {k: float(np.nanmean(v)) for k, v in scores.items()}, samples, all_metrics
