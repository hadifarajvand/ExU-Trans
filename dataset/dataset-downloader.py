from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from config import CONFIG  # noqa: E402
from dataset import DATASET_NAME, DATASET_SLUG, discover_dataset_root, write_dataset_audit, write_hdf5_schema_reports  # noqa: E402


def main() -> int:
    root = discover_dataset_root(CONFIG["DATASET_ROOT"])
    print("Dataset already expected locally. Searching for awsaf49/brats2020-training-data...")
    if root is None:
        print("STATUS = MISSING")
        print("Manual source URL: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data")
        return 0
    print(f"FOUND: {root}")
    print("Validating existing dataset...")
    write_hdf5_schema_reports(root, CONFIG["PREVIEW_DIR"] / "hdf5_schema_report.json", CONFIG["PREVIEW_DIR"] / "hdf5_schema_report.md")
    audit = write_dataset_audit(root, CONFIG["PREVIEW_DIR"] / "dataset_audit.json", CONFIG["PREVIEW_DIR"] / "dataset_audit.csv", metadata_csv=CONFIG["metadata_csv"])
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
    }
    (CONFIG["DATASET_ROOT"] / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
