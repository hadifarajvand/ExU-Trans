from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from config import CONFIG  # noqa: E402
from dataset import (  # noqa: E402
    DATASET_NAME,
    DATASET_SLUG,
    discover_dataset_root,
    generate_label_mapping_report,
    write_dataset_audit,
    write_hdf5_schema_reports,
    write_splits,
)


def run(args: list[str]) -> None:
    subprocess.run([str(PYTHON), *args], check=True, cwd=PROJECT_ROOT)


def detect_dataset_root() -> Path | None:
    root = discover_dataset_root(CONFIG["DATASET_ROOT"])
    if root is None and CONFIG["DATASET_DIR"].exists():
        root = CONFIG["DATASET_DIR"]
    return root


def collect_preflight() -> dict:
    dataset_root = detect_dataset_root()
    disk = shutil.disk_usage(PROJECT_ROOT)
    payload = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "python_executable": str(PYTHON),
        "venv_active": ".venv" in str(PYTHON),
        "dataset_name": DATASET_NAME,
        "dataset_slug": DATASET_SLUG,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "free_disk_bytes": disk.free,
        "total_disk_bytes": disk.total,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
    }
    return payload


def write_preflight() -> dict:
    out = CONFIG["PREVIEW_DIR"]
    out.mkdir(parents=True, exist_ok=True)
    payload = collect_preflight()
    (out / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def ensure_dataset_downloaded() -> None:
    root = detect_dataset_root()
    print("Dataset already expected locally. Searching for awsaf49/brats2020-training-data...")
    if root is None:
        print("STATUS = MISSING")
        print("Manual source URL: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data")
        return
    print(f"FOUND: {root}")
    print("Validating existing dataset...")
    write_hdf5_schema_reports(
        root,
        CONFIG["PREVIEW_DIR"] / "hdf5_schema_report.json",
        CONFIG["PREVIEW_DIR"] / "hdf5_schema_report.md",
    )
    audit = write_dataset_audit(
        root,
        CONFIG["PREVIEW_DIR"] / "dataset_audit.json",
        CONFIG["PREVIEW_DIR"] / "dataset_audit.csv",
        metadata_csv=CONFIG["metadata_csv"],
    )
    write_splits(root, CONFIG["MEASURED_DIR"] / "splits", seed=CONFIG["RANDOM_SEED"])
    generate_label_mapping_report(root, CONFIG["PREVIEW_DIR"] / "label_mapping_report.md")
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_slug": DATASET_SLUG,
        "source_format": "HDF5 slice representation",
        "source_status": "PRE-DOWNLOADED_LOCAL_DATASET",
        "download_performed_by_this_run": False,
        "local_root": str(root),
        "file_count": audit.get("file_count"),
        "unique_volume_count": audit.get("unique_volume_count"),
        "slice_count": audit.get("total_slice_count"),
        "metadata_csv": str(CONFIG["metadata_csv"]),
    }
    (CONFIG["DATASET_ROOT"] / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def run_smoke() -> None:
    run(["scripts/smoke_test.py"])


def run_train() -> None:
    run(["scripts/main.py"])


def run_reference() -> None:
    # Reference export only: never a measured/simulation result.
    run(["scripts/export.py"])


def run_compare(metrics_csv: str, target: str) -> None:
    run(["scripts/compare_with_paper.py", metrics_csv, "--target", target])


def run_evaluate() -> None:
    metrics_csv = CONFIG["MEASURED_DIR"] / "metrics" / "metrics_test.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Measured metrics not found: {metrics_csv}")
    run_compare(str(metrics_csv), "brats2020")


def run_export_measured(metrics_csv: str, output_root: str) -> None:
    run(["scripts/export_measured_paper_style.py", metrics_csv, "--output-root", output_root])


def run_budget(hours: float, epochs: int, baseline_seconds_per_batch: float) -> None:
    run([
        "scripts/budget_run.py",
        "--hours", str(hours),
        "--epochs", str(epochs),
        "--baseline-seconds-per-batch", str(baseline_seconds_per_batch),
    ])


def run_full() -> None:
    write_preflight()
    run(["-m", "pip", "check"])
    ensure_dataset_downloaded()
    run_smoke()
    run_train()
    run_evaluate()
    run_reference()
    metrics_csv = CONFIG["MEASURED_DIR"] / "metrics" / "metrics_test.csv"
    if metrics_csv.exists():
        run_compare(str(metrics_csv), "brats2020")
        run_export_measured(str(metrics_csv), str(CONFIG["MEASURED_DIR"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="ExU-Trans reproducibility pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight", help="Write system and repository preflight report")
    sub.add_parser("download-data", help="Locate/validate the existing local BraTS2020 HDF5 dataset; never auto-download")
    sub.add_parser("smoke", help="Run synthetic smoke test")
    sub.add_parser("train", help="Run measured real-data training path")
    sub.add_parser("evaluate", help="Compare measured test metrics against the BraTS2020 target")
    sub.add_parser("reference", help="Export published paper reference tables only; not simulation results")

    export_measured = sub.add_parser("export-measured", help="Generate paper-style figures/tables strictly from a measured metrics CSV")
    export_measured.add_argument("metrics_csv")
    export_measured.add_argument("--output-root", required=True)

    cmp_p = sub.add_parser("compare", help="Compare measured metrics CSV with paper target")
    cmp_p.add_argument("metrics_csv")
    cmp_p.add_argument("--target", choices=["table1", "brats2020", "brats2021"], default="table1")

    budget_p = sub.add_parser("budget-run", help="Run a deterministic genuine real-data subset within an approximate wall-time budget")
    budget_p.add_argument("--hours", type=float, default=2.0)
    budget_p.add_argument("--epochs", type=int, default=2)
    budget_p.add_argument("--baseline-seconds-per-batch", type=float, default=6.8)

    sub.add_parser("full", help="Run the end-to-end full real-data reproducibility scaffold")
    args = parser.parse_args()

    if args.mode == "preflight":
        write_preflight()
    elif args.mode == "download-data":
        ensure_dataset_downloaded()
    elif args.mode == "smoke":
        run_smoke()
    elif args.mode == "train":
        run_train()
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "reference":
        run_reference()
    elif args.mode == "export-measured":
        run_export_measured(args.metrics_csv, args.output_root)
    elif args.mode == "compare":
        run_compare(args.metrics_csv, args.target)
    elif args.mode == "budget-run":
        run_budget(args.hours, args.epochs, args.baseline_seconds_per_batch)
    elif args.mode == "full":
        run_full()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
