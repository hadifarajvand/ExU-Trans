"""Compare measured metrics with published ExU-Trans reference values.

This script never changes measured results and never fabricates missing values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

METRIC_MAP = {
    "dice": "DSC",
    "dsc": "DSC",
    "iou": "IoU",
    "hd95_mm": "HD",
    "hd_mm": "HD",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


def load_measured(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"No rows in measured metrics: {path}")
    aggregate = {}
    for source_name, paper_name in METRIC_MAP.items():
        values = []
        for row in rows:
            if source_name in row and row[source_name] not in ("", None):
                try:
                    values.append(float(row[source_name]))
                except ValueError:
                    pass
        if values:
            aggregate[paper_name] = mean(values)

    for key in ("DSC", "IoU", "Precision", "Recall", "F1"):
        if key in aggregate and 0 <= aggregate[key] <= 1.0:
            aggregate[key] *= 100.0
    return aggregate, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path, help="Measured per-case metrics CSV")
    parser.add_argument(
        "--target", choices=["table1", "brats2020", "brats2021"], default="table1"
    )
    parser.add_argument(
        "--paper-targets", type=Path, default=Path("reference/paper_targets.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/comparison/paper_alignment.json")
    )
    args = parser.parse_args()

    targets = json.loads(args.paper_targets.read_text(encoding="utf-8"))
    if args.target == "table1":
        published = targets["table_1_comparative"]["rows"]["ExU-Trans"]
    elif args.target == "brats2020":
        published = targets["table_3_dataset_wise"]["rows"]["BraTS 2020"]
    else:
        published = targets["table_3_dataset_wise"]["rows"]["BraTS 2021"]

    measured, n_rows = load_measured(args.metrics_csv)
    comparison = {}
    for metric, observed in measured.items():
        if metric not in published:
            continue
        reference = float(published[metric])
        comparison[metric] = {
            "measured": observed,
            "published": reference,
            "absolute_delta": observed - reference,
            "absolute_error": abs(observed - reference),
            "relative_error_percent": (
                None if reference == 0 else abs(observed - reference) / abs(reference) * 100.0
            ),
        }

    result = {
        "status": "COMPARISON_ONLY",
        "measured_source": str(args.metrics_csv),
        "measured_rows": n_rows,
        "paper_target": args.target,
        "comparison": comparison,
        "warnings": [
            "A numerical gap is not automatically an implementation bug.",
            "The paper contains internally inconsistent score sets; target table is explicit.",
            "HD/HD95 is comparable in millimetres only when physical voxel spacing is preserved.",
            "Slice-level metrics are not directly equivalent to case/volume-level metrics.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
