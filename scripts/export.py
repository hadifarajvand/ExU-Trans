"""Measured-output exporters. Reference paper values live under outputs/reference."""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ATTENTION_DIR,
    CHECKPOINT_DIR, MEASURED_DIR,
)


def setup_output_dirs():
    for d in [METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ATTENTION_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def export_metrics_table(all_metrics, dataset_type="validation"):
    df = pd.DataFrame([
        {"case_id": case_id, **metrics} for case_id, metrics in all_metrics.items()
    ])
    path = METRICS_DIR / f"metrics_{dataset_type}.csv"
    df.to_csv(path, index=False)
    print("Metrics saved:", path)
    return df


def export_comparison_chart(metrics_list, dataset_type="validation"):
    if not metrics_list:
        return None
    fields = [
        ("Dice", "dice"), ("IoU", "iou"), ("HD95 (grid px)", "hd95_px"),
        ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Measured {dataset_type} metrics (case-level)")
    for ax, (title, key) in zip(axes.flat, fields):
        values = [m[key] for m in metrics_list if key in m and np.isfinite(m[key])]
        if values:
            ax.hist(values, bins=min(15, max(3, len(values))), edgecolor="black", alpha=0.7)
        ax.set_title(f"{title} (mean: {np.nanmean(values):.4f})" if values else title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / f"metrics_{dataset_type}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_run_manifest(payload):
    MEASURED_DIR.mkdir(parents=True, exist_ok=True)
    path = MEASURED_DIR / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
