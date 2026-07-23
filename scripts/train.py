from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from losses import (
    segmentation_loss,
    to_pred,
    dice_score,
    hd95,
    iou_score,
    precision_score_np,
    recall_score_np,
    f1_score_np,
)


METRIC_FUNCS = {
    "dice": dice_score,
    "iou": iou_score,
    "precision": precision_score_np,
    "recall": recall_score_np,
    "f1": f1_score_np,
    "hd95_px": hd95,
}


def train_one_epoch(model, loader, optimizer, label_mode, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs, aux = model(images)
        loss, _ = segmentation_loss(outputs, masks, aux, label_mode=label_mode)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def _scalar_metrics(pred, target):
    return {name: fn(pred, target) for name, fn in METRIC_FUNCS.items()}


def _case_metrics(pred_volume, target_volume, label_mode):
    if label_mode != "regions":
        return _scalar_metrics(pred_volume, target_volume)

    if pred_volume.ndim != 4 or target_volume.ndim != 4:
        raise ValueError(
            f"regions evaluation expects [slices, channels, H, W], got "
            f"pred={pred_volume.shape}, target={target_volume.shape}"
        )
    if pred_volume.shape[1] != target_volume.shape[1]:
        raise ValueError("prediction/target channel mismatch during regions evaluation")

    result = {}
    per_region = {name: [] for name in METRIC_FUNCS}
    for region_idx in range(pred_volume.shape[1]):
        region_metrics = _scalar_metrics(pred_volume[:, region_idx], target_volume[:, region_idx])
        for metric_name, value in region_metrics.items():
            result[f"region_{region_idx}_{metric_name}"] = value
            per_region[metric_name].append(value)

    for metric_name, values in per_region.items():
        result[metric_name] = float(np.nanmean(values)) if values else float("nan")

    pred_union = np.any(pred_volume.astype(bool), axis=1)
    target_union = np.any(target_volume.astype(bool), axis=1)
    union_metrics = _scalar_metrics(pred_union, target_union)
    for metric_name, value in union_metrics.items():
        result[f"union_{metric_name}"] = value
    return result


def _batch_item(values, idx):
    if isinstance(values, torch.Tensor):
        value = values[idx]
        return value.item() if value.numel() == 1 else value
    if isinstance(values, (tuple, list)):
        return values[idx]
    return values


def evaluate(model, loader, label_mode, device):
    model.eval()
    by_case = {}
    sample_preview = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs, _ = model(images)
            pred = to_pred(outputs, label_mode=label_mode).detach().cpu().numpy().astype(np.uint8)
            target = masks.detach().cpu().numpy().astype(np.uint8)

            batch_size = int(images.shape[0])
            for batch_idx in range(batch_size):
                name = str(_batch_item(batch["volume_id"], batch_idx))
                slice_value = _batch_item(batch["slice_id"], batch_idx)
                idx = int(slice_value.item() if isinstance(slice_value, torch.Tensor) else slice_value)

                pred_item = pred[batch_idx]
                target_item = target[batch_idx]
                by_case.setdefault(name, {})[idx] = (pred_item, target_item)

                if len(sample_preview) < 8:
                    sample_preview.append(
                        (
                            images[batch_idx : batch_idx + 1].cpu(),
                            masks[batch_idx : batch_idx + 1].cpu(),
                            pred[batch_idx : batch_idx + 1],
                            name,
                            idx,
                            batch.get("metadata", {}),
                        )
                    )

    all_metrics = {}
    for name, slices in by_case.items():
        ordered = [slices[i] for i in sorted(slices)]
        pred_volume = np.stack([x[0] for x in ordered], axis=0)
        target_volume = np.stack([x[1] for x in ordered], axis=0)
        all_metrics[name] = _case_metrics(pred_volume, target_volume, label_mode)

    metric_keys = sorted({key for metrics in all_metrics.values() for key in metrics})
    summary = {
        key: float(np.nanmean([metrics.get(key, np.nan) for metrics in all_metrics.values()]))
        if all_metrics
        else float("nan")
        for key in metric_keys
    }
    return summary, sample_preview, all_metrics
