"""Explicit execution entry point.

No dependency installation and no synthetic fallback are performed silently.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(args):
    print("+", " ".join(map(str, args)))
    subprocess.run(args, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="ExU-Trans reproducibility pipeline")
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("smoke", help="Deterministic synthetic execution sanity check")
    sub.add_parser("reference", help="Export published paper values as REFERENCE_ONLY")
    sub.add_parser("train", help="Run the repository's measured training path")
    cmp_p = sub.add_parser("compare", help="Compare a measured metrics CSV with an explicit paper target")
    cmp_p.add_argument("metrics_csv")
    cmp_p.add_argument("--target", choices=["table1", "brats2020", "brats2021"], default="table1")
    args = p.parse_args()

    if args.mode == "smoke":
        run([sys.executable, "scripts/smoke_test.py"])
    elif args.mode == "reference":
        run([sys.executable, "scripts/export_paper_results.py"])
    elif args.mode == "train":
        run([sys.executable, "scripts/main.py"])
    elif args.mode == "compare":
        run([sys.executable, "scripts/compare_with_paper.py", args.metrics_csv, "--target", args.target])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
