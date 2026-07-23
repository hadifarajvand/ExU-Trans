"""BraTS2020 HDF5 discovery, validation, split safety, and 2D slice loading."""
from __future__ import annotations

import csv
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATASET_SLUG = "awsaf49/brats2020-training-data"
DATASET_NAME = "BraTS2020 Training Data"
DEFAULT_RELATIVE_ROOT = Path("dataset/fulldataset/BraTS2020_training_data/content/data")
DEFAULT_CSV = Path("dataset/fulldataset/BraTS20 Training Metadata.csv")

# The HDF5 tensor has four MRI channels, but exact channel provenance must come
# from the source conversion pipeline. Do not use this tuple as proof of order.
MODALITY_ORDER = ("FLAIR", "T1", "T1ce", "T2")


def _is_hdf5(path: Path) -> bool:
    return path.suffix.lower() in {".h5", ".hdf5"}


def discover_dataset_root(search_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    if search_root.exists():
        for path in search_root.rglob("*.h5"):
            if path.is_file():
                candidates.append(path.parent)
        for path in search_root.rglob("BraTS2020_training_data"):
            if path.is_dir():
                candidates.append(path / "content" / "data")
        for path in search_root.rglob("content/data"):
            if path.is_dir() and any(path.glob("*.h5")):
                candidates.append(path)
    candidates.append(DEFAULT_RELATIVE_ROOT)
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.h5")):
            return candidate.resolve()
    return None


@lru_cache(maxsize=1)
def _metadata_index(csv_path: str) -> Dict[Tuple[int, int], Dict[str, object]]:
    path = Path(csv_path)
    if not path.exists():
        return {}
    import pandas as pd

    frame = pd.read_csv(path)
    index: Dict[Tuple[int, int], Dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        volume = int(row["volume"])
        slice_idx = int(row["slice"])
        index[(volume, slice_idx)] = row
    return index


def resolve_metadata(csv_path: Optional[Path], volume: int, slice_idx: int) -> Dict[str, object]:
    if csv_path is None or not csv_path.exists():
        return {}
    row = _metadata_index(str(csv_path)).get((volume, slice_idx))
    return dict(row) if row else {}


def _parse_volume_slice(path: Path) -> Tuple[int, int]:
    # Expected filename: volume_146_slice_124.h5
    parts = path.stem.split("_")
    return int(parts[1]), int(parts[3])


def inspect_hdf5_file(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        image_arr = handle["image"][()]
        mask_arr = handle["mask"][()]
        attrs = {k: handle.attrs[k] for k in handle.attrs}
    volume, slice_idx = _parse_volume_slice(path)
    return {
        "path": str(path),
        "volume": volume,
        "slice": slice_idx,
        "keys": keys,
        "image_shape": list(image_arr.shape),
        "image_dtype": str(image_arr.dtype),
        "mask_shape": list(mask_arr.shape),
        "mask_dtype": str(mask_arr.dtype),
        "image_min": float(np.nanmin(image_arr)),
        "image_max": float(np.nanmax(image_arr)),
        "image_mean": float(np.nanmean(image_arr)),
        "image_std": float(np.nanstd(image_arr)),
        "mask_unique": sorted({int(x) for x in np.unique(mask_arr)}),
        "attrs": attrs,
    }


def write_hdf5_schema_reports(root: Path, out_json: Path, out_md: Path, sample_count: int = 3) -> Dict[str, object]:
    files = sorted(p for p in root.glob("*.h5") if p.is_file())
    if not files:
        sample_paths = []
    elif len(files) <= sample_count:
        sample_paths = files
    else:
        sample_paths = [files[0], files[len(files) // 2], files[-1]]
    reports = [inspect_hdf5_file(path) for path in sample_paths]
    payload = {
        "dataset_name": DATASET_NAME,
        "dataset_slug": DATASET_SLUG,
        "local_root": str(root.resolve()),
        "file_count": len(files),
        "samples": reports,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# HDF5 schema report",
        f"- dataset_name: {DATASET_NAME}",
        f"- dataset_slug: {DATASET_SLUG}",
        f"- local_root: {root.resolve()}",
        f"- file_count: {len(files)}",
        "",
    ]
    for report in reports:
        lines.extend([
            f"## {Path(report['path']).name}",
            f"- image_shape: {report['image_shape']}",
            f"- mask_shape: {report['mask_shape']}",
            f"- image_dtype: {report['image_dtype']}",
            f"- mask_dtype: {report['mask_dtype']}",
            f"- mask_unique: {report['mask_unique']}",
            f"- image_min: {report['image_min']}",
            f"- image_max: {report['image_max']}",
            f"- image_mean: {report['image_mean']}",
            f"- image_std: {report['image_std']}",
            "",
        ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return payload


def validate_sample_file(path: Path) -> Dict[str, object]:
    result = {"path": str(path), "valid": True, "issues": []}
    try:
        with h5py.File(path, "r") as handle:
            image = np.asarray(handle["image"])
            mask = np.asarray(handle["mask"])
        if image.ndim != 3 or mask.ndim != 3:
            result["valid"] = False
            result["issues"].append("unexpected_rank")
        if image.shape[:2] != mask.shape[:2]:
            result["valid"] = False
            result["issues"].append("shape_mismatch")
        if image.shape[-1] != 4:
            result["valid"] = False
            result["issues"].append("bad_channel_count")
        if mask.shape[-1] != 3:
            result["valid"] = False
            result["issues"].append("bad_mask_channel_count")
        if not np.isfinite(image).all():
            result["valid"] = False
            result["issues"].append("non_finite_image")
        if not np.isfinite(mask.astype(np.float32)).all():
            result["valid"] = False
            result["issues"].append("non_finite_mask")
        result["image_shape"] = list(image.shape)
        result["mask_shape"] = list(mask.shape)
        result["mask_unique"] = sorted({int(x) for x in np.unique(mask)})
        result["volume"], result["slice"] = _parse_volume_slice(path)
    except Exception as exc:
        result["valid"] = False
        result["issues"].append(type(exc).__name__)
    return result


def audit_dataset(root: Path, metadata_csv: Optional[Path] = None) -> Dict[str, object]:
    files = sorted(p for p in root.glob("*.h5") if p.is_file())
    rows = [validate_sample_file(path) for path in files]
    valid_rows = [row for row in rows if row.get("valid")]
    volumes = sorted({int(row["volume"]) for row in valid_rows})
    tumor_slices = 0
    background_only = 0
    duplicate_ids = 0
    seen = set()
    for row in valid_rows:
        key = (row["volume"], row["slice"])
        if key in seen:
            duplicate_ids += 1
        seen.add(key)
        if any(v != 0 for v in row.get("mask_unique", [])):
            tumor_slices += 1
        else:
            background_only += 1
    payload = {
        "dataset_name": DATASET_NAME,
        "dataset_slug": DATASET_SLUG,
        "local_root": str(root.resolve()),
        "file_count": len(files),
        "valid_samples": len(valid_rows),
        "invalid_samples": len(rows) - len(valid_rows),
        "unique_volume_count": len(volumes),
        "total_slice_count": len(valid_rows),
        "tumor_containing_slices": tumor_slices,
        "background_only_slices": background_only,
        "duplicate_slice_ids": duplicate_ids,
        "mask_unique_values_union": sorted({v for row in valid_rows for v in row.get("mask_unique", [])}),
        "rows": rows,
    }
    if metadata_csv and metadata_csv.exists():
        import pandas as pd

        meta = pd.read_csv(metadata_csv)
        payload["metadata_rows"] = len(meta)
        payload["metadata_columns"] = list(meta.columns)
    return payload


def write_dataset_audit(root: Path, out_json: Path, out_csv: Path, metadata_csv: Optional[Path] = None) -> Dict[str, object]:
    audit = audit_dataset(root, metadata_csv=metadata_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in audit.items() if k != "rows"}
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = audit.get("rows", [])
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def generate_label_mapping_report(root: Path, out_md: Path) -> Dict[str, object]:
    """Document only what the compact HDF5 masks actually prove."""
    files = sorted(p for p in root.glob("*.h5") if p.is_file())
    union = set()
    channel_positive = [0, 0, 0]
    checked = 0
    overlaps = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
    for path in files[: min(1000, len(files))]:
        with h5py.File(path, "r") as handle:
            mask = np.asarray(handle["mask"])
        union.update(int(x) for x in np.unique(mask))
        if mask.ndim == 3 and mask.shape[-1] == 3:
            binary = mask > 0
            for idx in range(3):
                channel_positive[idx] += int(binary[..., idx].sum())
            for pair in overlaps:
                overlaps[pair] += int(np.logical_and(binary[..., pair[0]], binary[..., pair[1]]).sum())
        checked += 1
    report = {
        "mask_unique_values": sorted(union),
        "mask_channels": 3,
        "channel_names": ["Region_0", "Region_1", "Region_2"],
        "channel_semantics_status": "UNRESOLVED_NAMES",
        "structural_note": "The project audit found the three HDF5 planes to be binary and mutually exclusive. Exact WT/TC/ET or source-label channel order is not claimed without provenance.",
        "sampled_files": checked,
        "sampled_positive_pixels_by_channel": channel_positive,
        "sampled_pairwise_overlap_pixels": {f"{a}_{b}": value for (a, b), value in overlaps.items()},
    }
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join([
            "# Label mapping report",
            f"- mask_unique_values: {report['mask_unique_values']}",
            "- mask_channels: 3",
            "- canonical_names: Region_0, Region_1, Region_2",
            "- semantic names: unresolved",
            f"- note: {report['structural_note']}",
        ]),
        encoding="utf-8",
    )
    return report


def make_volume_splits(volume_ids: Sequence[int], seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    ids = sorted(set(int(v) for v in volume_ids))
    train_ids, temp = train_test_split(ids, train_size=train_ratio, random_state=seed, shuffle=True)
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(sorted(temp), train_size=val_fraction, random_state=seed, shuffle=True)
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def write_splits(root: Path, out_dir: Path, seed: int = 42) -> Dict[str, object]:
    files = sorted(p for p in root.glob("*.h5") if p.is_file())
    volume_ids = sorted({_parse_volume_slice(path)[0] for path in files})
    train_ids, val_ids, test_ids = make_volume_splits(volume_ids, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        (out_dir / f"{name}_volumes.txt").write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")
    return {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
        "train_val_intersection": sorted(set(train_ids) & set(val_ids)),
        "train_test_intersection": sorted(set(train_ids) & set(test_ids)),
        "val_test_intersection": sorted(set(val_ids) & set(test_ids)),
    }


def label_map(mask_2d: np.ndarray, label_mode: str) -> np.ndarray:
    if label_mode == "whole_tumor":
        return (mask_2d > 0).astype(np.float32)
    if label_mode == "regions":
        if mask_2d.ndim == 3 and mask_2d.shape[-1] == 3:
            return mask_2d.transpose(2, 0, 1).astype(np.float32)
        raise ValueError("regions mode requires the explicit 3-channel HDF5 target; semantic remapping is intentionally not guessed")
    remapped = np.zeros_like(mask_2d, dtype=np.int64)
    remapped[mask_2d == 1] = 1
    remapped[mask_2d == 2] = 2
    remapped[mask_2d == 4] = 3
    return remapped


def _canonicalize_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim != 3:
        return mask_arr
    if mask_arr.shape[-1] == 3:
        return mask_arr
    if mask_arr.shape[0] == 3:
        return np.transpose(mask_arr, (1, 2, 0))
    return mask_arr


@lru_cache(maxsize=256)
def load_hdf5_sample(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        image = np.asarray(handle["image"], dtype=np.float32)
        mask = np.asarray(handle["mask"])
    return image, mask


class HDF5BratsSliceDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        image_size: int,
        label_mode: str,
        training: bool,
        debug_max_slices_per_case: Optional[int] = None,
        augment: bool = False,
        all_slices_for_evaluation: bool = True,
        metadata_csv: Optional[Path] = None,
    ):
        self.image_size = image_size
        self.label_mode = label_mode
        self.training = training
        self.augment = augment
        self.metadata_csv = metadata_csv
        self.items: List[Tuple[Path, int]] = []
        per_volume: Dict[int, List[Path]] = {}
        for path in files:
            volume, _ = _parse_volume_slice(path)
            per_volume.setdefault(volume, []).append(path)
        for volume in sorted(per_volume):
            paths = sorted(per_volume[volume], key=lambda p: _parse_volume_slice(p)[1])
            selected = paths[:debug_max_slices_per_case] if debug_max_slices_per_case is not None else paths
            self.items.extend((path, _parse_volume_slice(path)[1]) for path in selected)

    def __len__(self):
        return len(self.items)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image, mask = torch.flip(image, [2]), torch.flip(mask, [2])
        if random.random() < 0.5:
            image, mask = torch.flip(image, [1]), torch.flip(mask, [1])
        k = random.randint(0, 3)
        return torch.rot90(image, k, [1, 2]), torch.rot90(mask, k, [1, 2])

    def __getitem__(self, idx):
        path, slice_idx = self.items[idx]
        image, mask = load_hdf5_sample(str(path))
        volume, slice_no = _parse_volume_slice(path)
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_arr = _canonicalize_mask(mask)
        if self.label_mode == "regions":
            mask_t = torch.from_numpy(label_map(mask_arr, self.label_mode).copy())
        else:
            if mask_arr.ndim == 3 and mask_arr.shape[-1] == 3:
                mask_2d = mask_arr.any(axis=-1).astype(np.float32)
            else:
                mask_2d = mask_arr.squeeze()
            mask_t = torch.from_numpy(label_map(mask_2d, self.label_mode).copy())
        if image_t.shape[-2:] != (self.image_size, self.image_size):
            image_t = F.interpolate(image_t[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            mask_t = F.interpolate(mask_t[None].float(), size=(self.image_size, self.image_size), mode="nearest")[0]
        if self.training and self.augment:
            image_t, mask_t = self._augment(image_t, mask_t)
        mask_t = mask_t.float() if self.label_mode in {"whole_tumor", "regions"} else mask_t.long()
        metadata = resolve_metadata(self.metadata_csv, volume, slice_no)
        return {
            "image": image_t,
            "mask": mask_t,
            "volume_id": str(volume),
            "slice_id": slice_no,
            "metadata": metadata,
        }


def _read_volume_file(path: Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_loaders(config: Dict, summary: Dict):
    root = summary["dataset_root"]
    files = sorted(p for p in root.glob("*.h5") if p.is_file())
    volume_ids = sorted({_parse_volume_slice(path)[0] for path in files})
    if not volume_ids:
        empty = DataLoader(HDF5BratsSliceDataset([], config["image_size"], config["label_mode"], False), batch_size=1)
        return empty, empty, empty, empty

    split_dir = config["split_dir"]
    train_file = split_dir / "train_volumes.txt"
    val_file = split_dir / "validation_volumes.txt"
    test_file = split_dir / "test_volumes.txt"
    if train_file.exists() and val_file.exists() and test_file.exists():
        train_ids = _read_volume_file(train_file)
        val_ids = _read_volume_file(val_file)
        test_ids = _read_volume_file(test_file)
    else:
        train_ids, val_ids, test_ids = make_volume_splits(volume_ids, seed=config["RANDOM_SEED"])
        split_dir.mkdir(parents=True, exist_ok=True)
        train_file.write_text("\n".join(map(str, train_ids)) + "\n", encoding="utf-8")
        val_file.write_text("\n".join(map(str, val_ids)) + "\n", encoding="utf-8")
        test_file.write_text("\n".join(map(str, test_ids)) + "\n", encoding="utf-8")

    # Debug mode must subset VOLUMES, never the first N files/slices.
    if config["USE_DEBUG_SUBSET"]:
        n = config["DEBUG_NUM_CASES"]
        train_ids = train_ids[:n]
        val_ids = val_ids[:n]
        test_ids = test_ids[:n]

    def by_ids(ids: Sequence[int]) -> List[Path]:
        wanted = set(int(x) for x in ids)
        return [path for path in files if _parse_volume_slice(path)[0] in wanted]

    train_files = by_ids(train_ids)
    val_files = by_ids(val_ids)
    test_files = by_ids(test_ids)
    official_files = list(test_files)
    debug_n = config["debug_max_slices_per_case"] if config["USE_DEBUG_SUBSET"] else None
    metadata_csv = config.get("metadata_csv")

    train_ds = HDF5BratsSliceDataset(train_files, config["image_size"], config["label_mode"], True, debug_n, True, True, metadata_csv)
    val_ds = HDF5BratsSliceDataset(val_files, config["image_size"], config["label_mode"], False, debug_n, False, True, metadata_csv)
    test_ds = HDF5BratsSliceDataset(test_files, config["image_size"], config["label_mode"], False, debug_n, False, True, metadata_csv)
    official_ds = HDF5BratsSliceDataset(official_files, config["image_size"], config["label_mode"], False, debug_n, False, True, metadata_csv)
    return (
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]),
        DataLoader(official_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]),
    )
