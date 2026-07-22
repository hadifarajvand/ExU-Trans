"""Deterministic, data-free execution smoke test.

This proves that the model/loss/optimizer path executes. It is not a paper
reproduction and its synthetic metrics must never be presented as research results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from losses import segmentation_loss, dice_score, iou_score


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device=str(device), num_transformer_layers=4)
    model.train()

    image = torch.randn(1, 4, 128, 128, device=device)
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    synthetic = (((xx - 64) ** 2) / 24**2 + ((yy - 64) ** 2) / 18**2 <= 1).float()
    mask = synthetic[None, None].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    logits, aux = model(image)
    loss_before, parts = segmentation_loss(logits, mask, aux)
    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_after, _ = model(image)
        pred = (torch.sigmoid(logits_after) > 0.5).cpu().numpy().squeeze()
    target = mask.cpu().numpy().squeeze()

    result = {
        "status": "PASS",
        "mode": "synthetic_smoke_only",
        "device": str(device),
        "input_shape": list(image.shape),
        "output_shape": list(logits_after.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "transformer_layers": 4,
        "loss_before_step": float(loss_before.detach().cpu()),
        "loss_components": {k: float(v.cpu()) for k, v in parts.items()},
        "synthetic_dice_after_one_step": dice_score(pred, target),
        "synthetic_iou_after_one_step": iou_score(pred, target),
        "warning": "Synthetic smoke metrics are not paper reproduction results.",
    }
    out = Path("outputs/smoke/smoke_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
