"""Export published reference values from the paper target registry.

This replaces the old script that silently generated random metrics and mixed
hard-coded values with measured outputs. Nothing emitted by this script is a
simulation result.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> int:
    source = Path("reference/paper_targets.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    out_dir = Path("outputs/reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for table_name, table in data.items():
        if not table_name.startswith("table_") or not isinstance(table, dict) or "rows" not in table:
            continue
        rows = table["rows"]
        if not rows:
            continue
        keys = []
        for values in rows.values():
            for key in values:
                if key not in keys:
                    keys.append(key)
        path = out_dir / f"{table_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["label", *keys])
            writer.writeheader()
            for label, values in rows.items():
                writer.writerow({"label": label, **values})
        exported.append(str(path))

    manifest = {
        "status": "REFERENCE_ONLY",
        "source": str(source),
        "files": exported,
        "warning": "These values are transcribed from the paper and are not outputs of this repository's model.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
