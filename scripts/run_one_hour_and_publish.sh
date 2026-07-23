#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "ERROR: run this helper from the main branch."
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean. Commit/stash local changes first."
  git status --short
  exit 1
fi

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: .venv/bin/python not found."
  exit 1
fi

DATA_ROOT="$ROOT/dataset/fulldataset/BraTS2020_training_data/content/data"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: local HDF5 dataset not found at: $DATA_ROOT"
  exit 1
fi

mkdir -p outputs/measured_budget_60min

START_SHA="$(git rev-parse HEAD)"
echo "Starting from commit: $START_SHA"
echo "Python: $($PY --version)"

"$PY" -m pip check
"$PY" -m compileall scripts run_full_pipeline.py >/dev/null

set +e
"$PY" run_full_pipeline.py budget-run \
  --hours 1 \
  --epochs 2 \
  --baseline-seconds-per-batch 6.8 \
  2>&1 | tee outputs/measured_budget_60min/run_console.log
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ $RUN_STATUS -ne 0 ]]; then
  echo "ERROR: one-hour budget run failed with exit code $RUN_STATUS"
  exit $RUN_STATUS
fi

SOURCE="outputs/measured_budget_60min"
MANIFEST="$SOURCE/run_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: run completed without run_manifest.json"
  exit 1
fi

DEST="results/one_hour_latest"
rm -rf "$DEST"
mkdir -p "$DEST/metrics" "$DEST/figures" "$DEST/tables"

cp "$MANIFEST" "$DEST/run_manifest.json"
[[ -f "$SOURCE/run_plan.json" ]] && cp "$SOURCE/run_plan.json" "$DEST/run_plan.json"
[[ -f "$SOURCE/run_console.log" ]] && cp "$SOURCE/run_console.log" "$DEST/run_console.log"
find "$SOURCE/metrics" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' \) -exec cp {} "$DEST/metrics/" \; 2>/dev/null || true
find "$SOURCE/figures" -maxdepth 1 -type f -name '*.png' -exec cp {} "$DEST/figures/" \; 2>/dev/null || true
find "$SOURCE/tables" -maxdepth 1 -type f -name '*.csv' -exec cp {} "$DEST/tables/" \; 2>/dev/null || true

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path('results/one_hour_latest')
manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
summary = manifest.get('test_summary', {})
lines = [
    '# نتیجه اجرای واقعی یک‌ساعته ExU-Trans',
    '',
    '> این خروجی از اجرای واقعی روی زیرمجموعه deterministic از BraTS2020 ساخته شده و بازتولید کامل مقاله نیست.',
    '',
    f"- وضعیت: `{manifest.get('status')}`",
    f"- Device: `{manifest.get('device')}`",
    f"- Epochs completed: `{manifest.get('epochs_completed')}`",
    f"- Best epoch: `{manifest.get('best_epoch')}`",
    f"- Train volumes: `{len(manifest.get('train_volume_ids', []))}`",
    f"- Validation volumes: `{len(manifest.get('validation_volume_ids', []))}`",
    f"- Test volumes: `{len(manifest.get('test_volume_ids', []))}`",
    f"- Elapsed seconds: `{manifest.get('elapsed_seconds')}`",
    '',
    '## Test metrics',
    '',
    '| Metric | Value |',
    '|---|---:|',
]
for key in ('dice', 'iou', 'precision', 'recall', 'f1', 'hd95_px'):
    if key in summary:
        lines.append(f'| {key} | {summary[key]:.6f} |')
lines += [
    '',
    'جزئیات کامل در `run_manifest.json`، پوشه `metrics/`، `tables/` و `figures/` نگهداری شده است.',
]
(root / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

# Only publish compact result artifacts; dataset, .venv, checkpoints and raw outputs remain ignored.
git add "$DEST"

if git diff --cached --quiet; then
  echo "No publishable result changes detected."
  exit 0
fi

STAMP="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
git commit -m "results: publish one-hour real-data subset run ($STAMP)"
git push origin main

echo "PASS: one-hour real run completed and published to main."
echo "Results: $DEST"
