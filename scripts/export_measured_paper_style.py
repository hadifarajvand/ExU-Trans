"""Export paper-style figures/tables strictly from measured CSV metrics.

No synthetic fallback and no hard-coded measured values are permitted here.
Paper values are read only for a clearly-labelled comparison artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95_px"]
DISPLAY = {
    "dice": "Dice",
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "hd95_px": "HD95 (px)",
}


def _require_measured_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Measured metrics CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Measured metrics CSV is empty: {path}")
    missing = [metric for metric in BASE_METRICS if metric not in df.columns]
    if missing:
        raise ValueError(f"Measured metrics CSV missing required columns: {missing}")
    return df


def export_measured_outputs(metrics_csv: Path, output_root: Path) -> dict:
    df = _require_measured_csv(metrics_csv)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []

    # Table: measured summary statistics.
    summary_rows = []
    for metric in BASE_METRICS:
        values = pd.to_numeric(df[metric], errors="coerce")
        summary_rows.append({
            "metric": DISPLAY[metric],
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "n": int(values.notna().sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = tables_dir / "table_measured_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    artifacts.append(str(summary_path))

    # Figure: same metric families as the paper, measured case distributions.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Measured subset performance distributions")
    for ax, metric in zip(axes.flat, BASE_METRICS):
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not values.empty:
            ax.hist(values, bins=min(12, max(3, len(values))), edgecolor="black", alpha=0.75)
            ax.axvline(values.mean(), linestyle="--", linewidth=2, label=f"Mean={values.mean():.4f}")
            ax.legend()
        ax.set_title(DISPLAY[metric])
        ax.set_ylabel("Cases")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    metric_fig = figures_dir / "figure_measured_metric_distributions.png"
    fig.savefig(metric_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(metric_fig))

    # Per-region metrics, preserving anonymous region labels until provenance is proven.
    region_rows = []
    for region_idx in range(3):
        row = {"region": f"Region_{region_idx}"}
        found = False
        for metric in ["dice", "iou", "precision", "recall", "f1", "hd95_px"]:
            column = f"region_{region_idx}_{metric}"
            if column in df.columns:
                row[metric] = pd.to_numeric(df[column], errors="coerce").mean()
                found = True
        if found:
            region_rows.append(row)

    if region_rows:
        region_df = pd.DataFrame(region_rows)
        region_table = tables_dir / "table_measured_per_region.csv"
        region_df.to_csv(region_table, index=False)
        artifacts.append(str(region_table))

        plot_metrics = [metric for metric in ["dice", "iou", "precision", "recall", "f1"] if metric in region_df.columns]
        x = np.arange(len(region_df))
        width = 0.15
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, metric in enumerate(plot_metrics):
            ax.bar(x + idx * width, region_df[metric], width, label=DISPLAY[metric])
        ax.set_xticks(x + width * max(len(plot_metrics) - 1, 0) / 2)
        ax.set_xticklabels(region_df["region"])
        ax.set_ylabel("Measured score")
        ax.set_ylim(0, 1)
        ax.set_title("Measured performance by HDF5 target plane")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        region_fig = figures_dir / "figure_measured_per_region.png"
        fig.savefig(region_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(str(region_fig))

    # Clearly-labelled measured-vs-paper comparison for metric families with compatible scales.
    reference_path = Path("reference/paper_targets.json")
    if reference_path.exists():
        paper = json.loads(reference_path.read_text(encoding="utf-8"))
        target = paper["table_3_dataset_wise"]["rows"]["BraTS 2020"]
        mapping = {
            "dice": "DSC",
            "iou": "IoU",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }
        comparison_rows = []
        for measured_name, paper_name in mapping.items():
            measured = float(pd.to_numeric(df[measured_name], errors="coerce").mean() * 100.0)
            paper_value = float(target[paper_name])
            comparison_rows.append({
                "metric": paper_name,
                "measured_subset_percent": measured,
                "paper_brats2020_percent": paper_value,
                "absolute_gap_pp": measured - paper_value,
                "scope_note": "Subset measured run; not a full-paper reproduction",
            })
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_table = tables_dir / "table_measured_vs_paper.csv"
        comparison_df.to_csv(comparison_table, index=False)
        artifacts.append(str(comparison_table))

        x = np.arange(len(comparison_df))
        width = 0.36
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(x - width / 2, comparison_df["measured_subset_percent"], width, label="Measured subset")
        ax.bar(x + width / 2, comparison_df["paper_brats2020_percent"], width, label="Paper reference")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["metric"])
        ax.set_ylabel("Score (%)")
        ax.set_title("Measured subset vs published BraTS2020 reference")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        comparison_fig = figures_dir / "figure_measured_vs_paper.png"
        fig.savefig(comparison_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(str(comparison_fig))

    manifest = {
        "status": "MEASURED_ONLY",
        "source_metrics": str(metrics_csv),
        "artifacts": artifacts,
        "warnings": [
            "No synthetic fallback is allowed.",
            "No measured value is hard-coded.",
            "Region_0/1/2 are intentionally unnamed until channel provenance is proven.",
            "HD95 is reported in pixels unless physical spacing is available.",
            "Paper comparison is contextual only because a time-budgeted subset is not the full published experiment.",
        ],
    }
    manifest_path = output_root / "measured_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export paper-style artifacts from real measured metrics only")
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_measured_outputs(args.metrics_csv, args.output_root)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
