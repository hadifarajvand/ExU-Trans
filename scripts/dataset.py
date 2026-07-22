"""BraTS2020 discovery, deterministic splitting, and 2D slice loading."""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def _find_case_dirs(root: Path, prefix: str) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)) if root.exists() else []


def _case_files(case_dir: Path):
    files = {"flair": None, "t1": None, "t1ce": None, "t2": None, "seg": None}
    for path in case_dir.glob("*.nii*"):
        n = path.name.lower()
        if "_flair" in n:
            files["flair"] = path
        elif n.endswith("_t1.nii") or n.endswith("_t1.nii.gz"):
            files["t1"] = path
        elif "_t1ce" in n or "_t1gd" in n or "_t1g" in n:
            files["t1ce"] = path
        elif "_t2" in n:
            files["t2"] = path
        elif "_seg" in n:
            files["seg"] = path
    return files


@lru_cache(maxsize=24)
def load_nifti(path: str) -> np.ndarray:
    return np.transpose(nib.load(path).get_fdata().astype(np.float32), (2, 0, 1))


def zscore_non_background(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if mask.any():
        mean, std = float(volume[mask].mean()), float(volume[mask].std())
        std = 1.0 if std < 1e-6 else std
        out = volume.copy()
        out[mask] = (out[mask] - mean) / std
        out[~mask] = 0
        return out.astype(np.float32)
    return np.zeros_like(volume, dtype=np.float32)


def load_case(case_dir: Path):
    files = _case_files(case_dir)
    modalities = {}
    for key in ("flair", "t1", "t1ce", "t2"):
        if files[key] is None:
            raise FileNotFoundError(f"{case_dir.name}: missing modality {key}")
        modalities[key] = zscore_non_background(load_nifti(str(files[key])))
    mask = load_nifti(str(files["seg"])).astype(np.int64) if files["seg"] is not None else None
    return {"case_dir": case_dir, "modalities": modalities, "mask": mask, "files": files}


def label_map(mask_2d: np.ndarray, label_mode: str) -> np.ndarray:
    if label_mode == "whole_tumor":
        return (mask_2d > 0).astype(np.float32)
    remapped = np.zeros_like(mask_2d, dtype=np.int64)
    remapped[mask_2d == 1] = 1
    remapped[mask_2d == 2] = 2
    remapped[mask_2d == 4] = 3
    return remapped


def summarize_dataset(config: Dict):
    all_train_cases = _find_case_dirs(config["DATA_ROOT_TRAIN"], "BraTS20_Training_")
    official_val_cases = _find_case_dirs(config["DATA_ROOT_OFFICIAL_VAL"], "BraTS20_Validation_")
    if not all_train_cases:
        return {
            "train_cases": [], "val_cases": [], "test_cases": [],
            "official_val_cases": official_val_cases,
            "train_count": 0, "val_count": 0, "test_count": 0,
            "official_val_count": len(official_val_cases),
        }
    train_cases, temp = train_test_split(
        all_train_cases, train_size=config["TRAIN_RATIO"],
        random_state=config["RANDOM_SEED"], shuffle=True,
    )
    val_fraction = config["VAL_RATIO"] / (config["VAL_RATIO"] + config["TEST_RATIO"])
    val_cases, test_cases = train_test_split(
        temp, train_size=val_fraction,
        random_state=config["RANDOM_SEED"], shuffle=True,
    )
    result = {
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "test_cases": sorted(test_cases),
        "official_val_cases": sorted(official_val_cases),
    }
    for key in ("train", "val", "test", "official_val"):
        result[f"{key}_count"] = len(result[f"{key}_cases"])
    print("Dataset Summary:", {k: v for k, v in result.items() if k.endswith("_count")})
    return result


class BratsSliceDataset(Dataset):
    def __init__(
        self,
        case_dirs: Sequence[Path],
        image_size: int,
        label_mode: str,
        training: bool,
        debug_max_slices_per_case: Optional[int] = None,
        augment: bool = False,
        allow_missing_seg: bool = False,
        all_slices_for_evaluation: bool = True,
    ):
        self.image_size = image_size
        self.label_mode = label_mode
        self.training = training
        self.augment = augment
        self.allow_missing_seg = allow_missing_seg
        self.items = []
        for case_dir in case_dirs:
            files = _case_files(case_dir)
            if files["seg"] is not None:
                mask = load_nifti(str(files["seg"])).astype(np.int64)
                if training or not all_slices_for_evaluation:
                    selected = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0].tolist()
                    selected = selected or [mask.shape[0] // 2]
                else:
                    selected = list(range(mask.shape[0]))
            elif allow_missing_seg and files["flair"] is not None:
                selected = list(range(load_nifti(str(files["flair"])).shape[0]))
            else:
                continue
            if debug_max_slices_per_case is not None:
                selected = selected[:debug_max_slices_per_case]
            self.items.extend((case_dir, int(s)) for s in selected)

    def __len__(self):
        return len(self.items)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image, mask = torch.flip(image, [2]), torch.flip(mask, [1])
        if random.random() < 0.5:
            image, mask = torch.flip(image, [1]), torch.flip(mask, [0])
        k = random.randint(0, 3)
        return torch.rot90(image, k, [1, 2]), torch.rot90(mask, k, [0, 1])

    def __getitem__(self, idx):
        case_dir, slice_idx = self.items[idx]
        sample = load_case(case_dir)
        mods = sample["modalities"]
        image = np.stack([
            mods["flair"][slice_idx], mods["t1"][slice_idx],
            mods["t1ce"][slice_idx], mods["t2"][slice_idx],
        ], axis=0)
        raw_mask = sample["mask"]
        mask2d = np.zeros(image.shape[-2:], np.float32) if raw_mask is None else label_map(raw_mask[slice_idx], self.label_mode)
        image_t = torch.from_numpy(image.copy()).float()
        mask_t = torch.from_numpy(mask2d.copy())
        if image_t.shape[-2:] != (self.image_size, self.image_size):
            image_t = F.interpolate(
                image_t[None], size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )[0]
            mask_t = F.interpolate(
                mask_t[None, None].float(), size=(self.image_size, self.image_size),
                mode="nearest",
            )[0, 0]
        if self.training and self.augment:
            image_t, mask_t = self._augment(image_t, mask_t)
        mask_t = mask_t[None].float() if self.label_mode == "whole_tumor" else mask_t.long()
        return image_t, mask_t, case_dir.name, slice_idx


def build_loaders(config: Dict, summary: Dict):
    def maybe_subset(items):
        return items[:config["DEBUG_NUM_CASES"]] if config["USE_DEBUG_SUBSET"] else items

    train_cases = maybe_subset(summary["train_cases"])
    val_cases = maybe_subset(summary["val_cases"])
    test_cases = maybe_subset(summary["test_cases"])
    official = maybe_subset(summary["official_val_cases"])
    debug_n = config["debug_max_slices_per_case"] if config["USE_DEBUG_SUBSET"] else None

    train_ds = BratsSliceDataset(
        train_cases, config["image_size"], config["label_mode"], True,
        debug_n, True, False, False,
    )
    val_ds = BratsSliceDataset(
        val_cases, config["image_size"], config["label_mode"], False,
        debug_n, False, False, True,
    )
    test_ds = BratsSliceDataset(
        test_cases, config["image_size"], config["label_mode"], False,
        debug_n, False, False, True,
    )
    official_ds = BratsSliceDataset(
        official, config["image_size"], config["label_mode"], False,
        debug_n, False, True, True,
    )
    return (
        DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]),
        DataLoader(official_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"]),
    )
