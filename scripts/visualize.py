"""Visualization utilities."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from losses import to_pred


@torch.no_grad()
def show_prediction(sample_tuple, model, device, label_mode="whole_tumor"):
    """Visualize prediction vs ground truth."""
    images, masks, case_id, slice_idx = sample_tuple
    images = images.unsqueeze(0).to(device)
    logits, aux = model(images)
    pred = to_pred(logits, label_mode).cpu().numpy()[0]
    image = images.cpu().numpy()[0]
    mask = masks.numpy()[0]

    if label_mode != "whole_tumor":
        mask = (mask > 0).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("FLAIR")
    axes[1].imshow(mask.squeeze(), cmap="magma")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred.squeeze(), cmap="magma")
    axes[2].set_title("Prediction")
    axes[3].imshow(image[0], cmap="gray")
    axes[3].imshow(pred.squeeze(), cmap="Reds", alpha=0.35)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"{case_id} | slice {slice_idx}")
    plt.tight_layout()
    plt.show()
    return pred, aux


def show_sample(sample):
    """Visualize training sample."""
    image_t, mask_t, case_id, slice_idx = sample
    image, mask = image_t.numpy(), mask_t.numpy()

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    for i, title in enumerate(['FLAIR', 'T1', 'T1ce', 'T2']):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    axes[4].imshow(mask.squeeze(), cmap='magma')
    axes[4].set_title('Mask')
    axes[4].axis('off')

    axes[5].imshow(image[0], cmap='gray')
    axes[5].imshow(mask.squeeze(), cmap='Reds', alpha=0.35)
    axes[5].set_title('Overlay')
    axes[5].axis('off')

    plt.suptitle(case_id + ' | slice ' + str(slice_idx))
    plt.tight_layout()
    plt.show()
