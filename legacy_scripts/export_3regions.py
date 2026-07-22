"""
Export functions for 3-region evaluation results
Creates tables and figures matching the paper format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def create_summary_table(df_3regions: pd.DataFrame) -> str:
    """Create a summary table like Table 4 in the paper

    Args:
        df_3regions: DataFrame with columns [case_id, region, dice, iou, precision, recall, f1, hd95]

    Returns:
        Formatted table string
    """
    summary = []
    summary.append("Table: Segmentation Performance by Tumor Region")
    summary.append("=" * 90)
    summary.append(f"{'Region':<12} {'Dice':<12} {'IoU':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'HD95 (mm)':<12}")
    summary.append("-" * 90)

    for region in ['WT', 'TC', 'ET']:
        region_data = df_3regions[df_3regions['region'] == region]
        if len(region_data) == 0:
            continue

        metrics_dict = {}
        for metric in ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']:
            valid_vals = region_data[metric].dropna()
            if len(valid_vals) > 0:
                mean_val = valid_vals.mean()
                std_val = valid_vals.std()
                metrics_dict[metric] = (mean_val, std_val)

        line = f"{region:<12}"
        for metric in ['dice', 'iou', 'precision', 'recall', 'f1']:
            mean_val, std_val = metrics_dict.get(metric, (0, 0))
            line += f"{mean_val:.4f}±{std_val:.4f}  "

        if 'hd95' in metrics_dict:
            mean_val, std_val = metrics_dict['hd95']
            line += f"{mean_val:.2f}±{std_val:.2f}"

        summary.append(line)

    summary.append("=" * 90)
    return "\n".join(summary)


def export_summary_table_png(df_3regions: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Export summary table as PNG image (publication-quality)

    Args:
        df_3regions: DataFrame with 3-region metrics
        output_path: Where to save the PNG
        dpi: DPI for export (300 for publication quality)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for table
    table_data = []
    for region in ['WT', 'TC', 'ET']:
        region_data = df_3regions[df_3regions['region'] == region]
        if len(region_data) == 0:
            continue

        row = [region]
        for metric in ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']:
            valid_vals = region_data[metric].dropna()
            if len(valid_vals) > 0:
                mean_val = valid_vals.mean()
                std_val = valid_vals.std()

                if metric == 'hd95':
                    row.append(f"{mean_val:.2f}±{std_val:.2f}")
                else:
                    row.append(f"{mean_val:.4f}±{std_val:.4f}")
            else:
                row.append("N/A")

        table_data.append(row)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')

    columns = ['Region', 'Dice', 'IoU', 'Precision', 'Recall', 'F1', 'HD95 (mm)']
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title('Table: Segmentation Performance by Tumor Region', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Summary table saved to: {output_path}")
    plt.close()


def export_metric_distributions(df_3regions: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Create distribution plots for each metric by region

    Args:
        df_3regions: DataFrame with metrics
        output_path: Where to save the PNG
        dpi: DPI for export
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metric Distributions by Tumor Region', fontsize=16, fontweight='bold')

    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']
    regions = ['WT', 'TC', 'ET']
    colors = {'WT': '#1f77b4', 'TC': '#ff7f0e', 'ET': '#2ca02c'}

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for region in regions:
            region_data = df_3regions[df_3regions['region'] == region]
            if len(region_data) > 0:
                values = region_data[metric].dropna()
                if len(values) > 0:
                    ax.hist(values, bins=15, alpha=0.6, label=region, color=colors[region])

        ax.set_xlabel('Score' if metric != 'hd95' else 'Distance (mm)')
        ax.set_ylabel('Frequency')
        ax.set_title(metric.upper())
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Metric distributions saved to: {output_path}")
    plt.close()


def export_region_comparison(df_3regions: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Create bar chart comparing metrics across regions

    Args:
        df_3regions: DataFrame with metrics
        output_path: Where to save the PNG
        dpi: DPI for export
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    regions = ['WT', 'TC', 'ET']
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    means = {}
    stds = {}

    for region in regions:
        region_data = df_3regions[df_3regions['region'] == region]
        means[region] = []
        stds[region] = []

        for metric in metrics:
            valid_vals = region_data[metric].dropna()
            if len(valid_vals) > 0:
                means[region].append(valid_vals.mean())
                stds[region].append(valid_vals.std())
            else:
                means[region].append(0)
                stds[region].append(0)

    colors = {'WT': '#1f77b4', 'TC': '#ff7f0e', 'ET': '#2ca02c'}

    for i, region in enumerate(regions):
        offset = width * (i - 1)
        ax.bar(x + offset, means[region], width, label=region, color=colors[region],
               yerr=stds[region], capsize=5, alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Segmentation Performance by Region', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Region comparison chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Export functions for 3-region evaluation loaded")
