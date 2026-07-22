"""Export metrics and visualizations."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ATTENTION_DIR


def setup_output_dirs():
    """Create output directories."""
    for d in [METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ATTENTION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created:")
    for d in [METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ATTENTION_DIR]:
        print(f"  {d}")


def export_metrics_table(all_metrics, dataset_type="validation"):
    """Export metrics table as CSV."""
    metrics_list = [{"case_id": case_id, **metrics} for case_id, metrics in all_metrics.items()]
    df = pd.DataFrame(metrics_list)
    csv_path = METRICS_DIR / f"metrics_{dataset_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")
    return df


def export_comparison_chart(metrics_list):
    """Export metrics comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ExU-Trans Performance Metrics", fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ("Dice", "dice", (0, 0)),
        ("IoU", "iou", (0, 1)),
        ("HD95", "hd95", (0, 2)),
        ("Precision", "precision", (1, 0)),
        ("Recall", "recall", (1, 1)),
        ("F1", "f1", (1, 2))
    ]

    for title, metric, (row, col) in metrics_to_plot:
        ax = axes[row, col]
        values = [m[metric] for m in metrics_list]
        ax.hist(values, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"{title} (mean: {np.nanmean(values):.4f})")
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
