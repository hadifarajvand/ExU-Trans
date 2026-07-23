"""Losses and segmentation metrics.

HD95 is returned in pixels unless an explicit physical spacing is supplied.
Do not label an uncalibrated HD95 value as millimetres.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_loss_binary(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def multiclass_dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()

def multilabel_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(2, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def trait_guided_alignment_loss(attr_map, mask):
    # attr_map is already sigmoid-normalized to [0, 1]. Keep the target binary.
    return F.mse_loss(attr_map, (mask > 0).float())


def boundary_aware_loss(logits, targets):
    pred = torch.sigmoid(logits)
    targets_f = targets.float()
    pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_grad_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_grad_x = targets_f[:, :, 1:, :] - targets_f[:, :, :-1, :]
    tgt_grad_y = targets_f[:, :, :, 1:] - targets_f[:, :, :, :-1]
    return (F.mse_loss(pred_grad_x, tgt_grad_x) + F.mse_loss(pred_grad_y, tgt_grad_y)) / 2


def segmentation_loss(
    logits,
    target,
    aux,
    label_mode="whole_tumor",
    pixel_weight=1.0,
    align_weight=0.05,
    boundary_weight=0.05,
):
    """Paper-inspired TGOF approximation: pixel + alignment + boundary terms."""
    if label_mode == "whole_tumor":
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = dice_loss_binary(logits, target)
        pixel = bce + dice
        align = trait_guided_alignment_loss(aux["attr_map"], target)
        boundary = boundary_aware_loss(logits, target)
        total = pixel_weight * pixel + align_weight * align + boundary_weight * boundary
        return total, {
            "pixel": pixel.detach(), "bce": bce.detach(), "dice": dice.detach(),
            "align": align.detach(), "boundary": boundary.detach(),
        }

    if label_mode == "regions":
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = multilabel_dice_loss(logits, target)
        total = pixel_weight * (bce + dice) + align_weight * trait_guided_alignment_loss(aux["attr_map"], target[:, :1])
        return total, {"pixel": (bce + dice).detach(), "bce": bce.detach(), "dice": dice.detach(), "align": torch.tensor(0.0), "boundary": torch.tensor(0.0)}

    ce = F.cross_entropy(logits, target.long())
    dice = multiclass_dice_loss(logits, target.long())
    whole = (target > 0).float().unsqueeze(1)
    pixel = ce + dice
    align = trait_guided_alignment_loss(aux["attr_map"], whole)
    probs = torch.softmax(logits, dim=1)
    foreground_prob = probs[:, 1:].sum(dim=1, keepdim=True).clamp(1e-6, 1 - 1e-6)
    binary_logits = torch.logit(foreground_prob)
    boundary = boundary_aware_loss(binary_logits, whole)
    total = pixel_weight * pixel + align_weight * align + boundary_weight * boundary
    return total, {
        "pixel": pixel.detach(), "ce": ce.detach(), "dice": dice.detach(),
        "align": align.detach(), "boundary": boundary.detach(),
    }


def to_pred(logits, label_mode="whole_tumor"):
    if label_mode == "whole_tumor":
        return (torch.sigmoid(logits) > 0.5).long()
    return torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)


def dice_score(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))


def precision_score_np(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score_np(pred, target, eps=1e-6):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float((tp + eps) / (tp + fn + eps))


def f1_score_np(pred, target, eps=1e-6):
    p, r = precision_score_np(pred, target, eps), recall_score_np(pred, target, eps)
    return float((2 * p * r + eps) / (p + r + eps))


def hd95(pred, target, spacing=None):
    """Symmetric 95th percentile Hausdorff distance.

    spacing=None means output units are pixels/grid units. For physical units,
    pass the spacing corresponding to the evaluated array dimensions.
    """
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    pred_surf = pred ^ binary_erosion(pred)
    tgt_surf = target ^ binary_erosion(target)
    dist_to_target = distance_transform_edt(~target, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred, sampling=spacing)
    dist_a = dist_to_target[pred_surf]
    dist_b = dist_to_pred[tgt_surf]
    return float(np.percentile(np.concatenate([dist_a, dist_b]), 95)) if dist_a.size and dist_b.size else float("nan")
