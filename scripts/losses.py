"""Loss functions and metrics for segmentation."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_loss_binary(logits, targets, eps=1e-6):
    """Binary dice loss."""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def multiclass_dice_loss(logits, targets, eps=1e-6):
    """Multiclass dice loss."""
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def trait_guided_alignment_loss(attr_map, mask, eps=1e-6):
    """Align attention maps with tumor regions."""
    mask_normalized = torch.sigmoid(mask.float())
    return F.mse_loss(attr_map, mask_normalized)


def boundary_aware_loss(logits, targets, eps=1e-6):
    """Penalize gradient differences at boundaries."""
    pred = torch.sigmoid(logits)
    targets_f = targets.float()
    pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_grad_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_grad_x = targets_f[:, :, 1:, :] - targets_f[:, :, :-1, :]
    tgt_grad_y = targets_f[:, :, :, 1:] - targets_f[:, :, :, :-1]
    loss_x = F.mse_loss(pred_grad_x, tgt_grad_x)
    loss_y = F.mse_loss(pred_grad_y, tgt_grad_y)
    return (loss_x + loss_y) / 2


def segmentation_loss(logits, target, aux, label_mode="whole_tumor",
                      attr_weight=0.1, align_weight=0.05, boundary_weight=0.05):
    """Trait-Guided Optimization Function (TGOF)."""
    if label_mode == "whole_tumor":
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = dice_loss_binary(logits, target)
        align = trait_guided_alignment_loss(aux["attr_map"], target)
        boundary = boundary_aware_loss(logits, target)
        total_loss = bce + dice + attr_weight * align + align_weight * align + boundary_weight * boundary
        return total_loss, {"bce": bce.detach(), "dice": dice.detach(), "attr": align.detach(), "boundary": boundary.detach()}

    ce = F.cross_entropy(logits, target.long())
    dice = multiclass_dice_loss(logits, target.long())
    whole = (target > 0).float().unsqueeze(1)
    attr = F.binary_cross_entropy(aux["attr_map"], whole)
    align = trait_guided_alignment_loss(aux["attr_map"], whole)
    boundary = boundary_aware_loss(logits, whole)
    total_loss = ce + dice + attr_weight * attr + align_weight * align + boundary_weight * boundary
    return total_loss, {"ce": ce.detach(), "dice": dice.detach(), "attr": attr.detach(), "align": align.detach(), "boundary": boundary.detach()}


def to_pred(logits, label_mode="whole_tumor"):
    """Convert logits to predictions."""
    if label_mode == "whole_tumor":
        return (torch.sigmoid(logits) > 0.5).long()
    else:
        return torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)


def dice_score(pred, target, eps=1e-6):
    """Compute Dice coefficient."""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred, target, eps=1e-6):
    """Compute IoU (Jaccard index)."""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))


def precision_score_np(pred, target, eps=1e-6):
    """Compute precision."""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score_np(pred, target, eps=1e-6):
    """Compute recall."""
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()
    return float((tp + eps) / (tp + fn + eps))


def f1_score_np(pred, target, eps=1e-6):
    """Compute F1 score."""
    p, r = precision_score_np(pred, target, eps), recall_score_np(pred, target, eps)
    return float((2 * p * r + eps) / (p + r + eps))


def hd95(pred, target):
    """Compute Hausdorff distance 95th percentile."""
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    pred_surf = pred ^ binary_erosion(pred)
    tgt_surf = target ^ binary_erosion(target)
    dist_a = distance_transform_edt(~target)[pred_surf]
    dist_b = distance_transform_edt(~pred)[tgt_surf]
    return float(np.percentile(np.concatenate([dist_a, dist_b]), 95)) if dist_a.size and dist_b.size else float("nan")
