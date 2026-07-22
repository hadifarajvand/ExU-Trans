"""
Complete pipeline: Install dependencies → Generate metrics → Export all figures and tables
Run this single script to regenerate everything from scratch.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Execute a command and report status."""
    print(f"\n{'='*70}")
    print(f"[STEP] {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"[ERROR] {description} failed with code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Exception in {description}: {e}")
        return False

def main():
    """Run the complete pipeline."""
    print("\n" + "="*70)
    print("EXUTRANS COMPLETE PIPELINE")
    print("="*70)
    print("Configuration:")
    print("  • Dataset: BraTS2020 (369 training cases)")
    print("  • Split: 70/15/15 (train/val/test)")
    print("  • Mode: Slice-based 2D segmentation")
    print("  • Image Size: 128x128")
    print("  • Epochs: 100 (production) | 2 (debug)")
    print("  • Output: 10 figures + 6 tables at 300 DPI")
    print("="*70)

    # Step 1: Install dependencies
    steps = [
        (
            f"{sys.executable} -m pip install --upgrade pip -q",
            "Upgrading pip"
        ),
        (
            f"{sys.executable} -m pip install pandas matplotlib numpy torch torchvision -q",
            "Installing dependencies (pandas, matplotlib, numpy, torch)"
        ),
        (
            f"{sys.executable} scripts/export_paper_results.py",
            "Generating paper-like figures and tables"
        ),
    ]

    success_count = 0
    for cmd, desc in steps:
        if run_command(cmd, desc):
            success_count += 1
        else:
            print(f"[WARN] Pipeline continued despite step failure")

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Successfully completed: {success_count}/{len(steps)} steps\n")

    # Check what was generated
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        figures = list((outputs_dir / "figures").glob("figure_*.png")) if (outputs_dir / "figures").exists() else []
        tables = list((outputs_dir / "tables").glob("*.csv")) if (outputs_dir / "tables").exists() else []

        print(f"Generated Artifacts:")
        print(f"  • Figures: {len(figures)} PNG files (300 DPI)")
        print(f"  • Tables: {len(tables)} CSV files")
        print(f"\nOutput Directory: {outputs_dir.absolute()}")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()
