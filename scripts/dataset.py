"""Dataset utilities for BraTS2020 data loading and preprocessing."""

import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split

from config import CONFIG


def _find_case_dirs(root: Path, prefix: str) -> List[Path]:
    """Find all case directories matching prefix."""
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]) if root.exists() else []


def _case_files(case_dir: Path) -> Dict[str, Optional[Path]]:
    """Extract file paths for modalities and segmentation."""
    files = {'flair': None, 't1': None, 't1ce': None, 't2': None, 'seg': None}
    for path in case_dir.glob('*.nii*'):
        n = path.name.lower()
        if '_flair' in n:
            files['flair'] = path
        elif n.endswith('_t1.nii') or n.endswith('_t1.nii.gz'):
            files['t1'] = path
        elif '_t1ce' in n or '_t1gd' in n or '_t1g' in n:
            files['t1ce'] = path
        elif '_t2' in n:
            files['t2'] = path
        elif '_seg' in n:
            files['seg'] = path
    return files


@lru_cache(maxsize=16)
def load_nifti(path: str) -> np.ndarray:
    """Load NIfTI file and transpose to (Z, H, W)."""
    return np.transpose(nib.load(path).get_fdata().astype(np.float32), (2, 0, 1))


def zscore_non_background(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize non-background voxels."""
    mask = volume > 0
    if mask.any():
        mean, std = float(volume[mask].mean()), float(volume[mask].std())
        std = 1.0 if std < 1e-6 else std
        volume = (volume - mean) / std
    else:
        volume = volume - volume.mean()
    return volume.astype(np.float32)


def load_case(case_dir: Path) -> Dict[str, object]:
    """Load all modalities and segmentation for a case."""
    files = _case_files(case_dir)
    modalities = {}
    for key in ('flair', 't1', 't1ce', 't2'):
        if files[key] is None:
            raise FileNotFoundError('missing ' + key)
        modalities[key] = zscore_non_background(load_nifti(str(files[key])))
    mask = load_nifti(str(files['seg'])).astype(np.int64) if files['seg'] is not None else None
    return {'case_dir': case_dir, 'modalities': modalities, 'mask': mask, 'files': files}


def make_binary_whole_tumor(mask: np.ndarray) -> np.ndarray:
    """Convert multi-class segmentation to binary whole tumor."""
    return (mask > 0).astype(np.float32)


def make_subregion_classes(mask: np.ndarray) -> np.ndarray:
    """Remap tumor subregions."""
    remapped = np.zeros_like(mask, dtype=np.int64)
    remapped[mask == 1] = 1
    remapped[mask == 2] = 2
    remapped[mask == 4] = 3
    return remapped


def extract_tumor_regions(seg_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract WT, TC, ET regions."""
    wt = ((seg_mask == 1) | (seg_mask == 2) | (seg_mask == 4)).astype(np.float32)
    tc = ((seg_mask == 1) | (seg_mask == 4)).astype(np.float32)
    et = (seg_mask == 4).astype(np.float32)
    return {'WT': wt, 'TC': tc, 'ET': et}


def label_map(mask_2d: np.ndarray, label_mode: str) -> np.ndarray:
    """Apply label mode transformation."""
    if label_mode == 'whole_tumor':
        return make_binary_whole_tumor(mask_2d)
    else:
        return make_subregion_classes(mask_2d)


def summarize_dataset(config: Dict) -> Dict[str, object]:
    """Discover and split dataset."""
    train_root = config['DATA_ROOT_TRAIN']
    official_val_root = config['DATA_ROOT_OFFICIAL_VAL']
    all_train_cases = _find_case_dirs(train_root, 'BraTS20_Training_')
    official_val_cases = _find_case_dirs(official_val_root, 'BraTS20_Validation_')

    if len(all_train_cases) == 0:
        print('[WARNING] No training cases found')
        return {
            'train_cases': [], 'val_cases': [], 'test_cases': [],
            'official_val_cases': [], 'train_count': 0, 'val_count': 0,
            'test_count': 0, 'official_val_count': 0
        }

    train_cases, temp_cases = train_test_split(
        all_train_cases, train_size=config['TRAIN_RATIO'],
        random_state=config['RANDOM_SEED'], shuffle=True
    )
    val_ratio = config['VAL_RATIO'] / (config['VAL_RATIO'] + config['TEST_RATIO'])
    val_cases, test_cases = train_test_split(
        temp_cases, train_size=val_ratio,
        random_state=config['RANDOM_SEED'], shuffle=True
    )

    train_cases = sorted(train_cases)
    val_cases = sorted(val_cases)
    test_cases = sorted(test_cases)
    official_val_cases = sorted(official_val_cases)

    summary = {
        'train_cases': train_cases, 'val_cases': val_cases,
        'test_cases': test_cases, 'official_val_cases': official_val_cases
    }
    summary['train_count'] = len(train_cases)
    summary['val_count'] = len(val_cases)
    summary['test_count'] = len(test_cases)
    summary['official_val_count'] = len(official_val_cases)

    print('Dataset Summary:')
    print(f'Training: {len(all_train_cases)} split as train={summary["train_count"]} val={summary["val_count"]} test={summary["test_count"]}')
    print(f'Official validation: {summary["official_val_count"]}')
    return summary


class BratsSliceDataset(Dataset):
    """2D slice-based dataset for BraTS2020."""

    def __init__(self, case_dirs: Sequence[Path], image_size: int, label_mode: str,
                 training: bool, debug_max_slices_per_case: Optional[int] = None,
                 augment: bool = False, allow_missing_seg: bool = False):
        self.image_size = image_size
        self.label_mode = label_mode
        self.training = training
        self.augment = augment
        self.allow_missing_seg = allow_missing_seg
        self.items = []

        for case_dir in case_dirs:
            files = _case_files(case_dir)
            seg_path = files['seg']
            if seg_path is None:
                if allow_missing_seg:
                    self.items.append((case_dir, 0, True))
                continue

            mask = load_nifti(str(seg_path)).astype(np.int64)
            tumor_slices = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0].tolist()
            selected = tumor_slices if tumor_slices else [mask.shape[0] // 2]
            if debug_max_slices_per_case is not None:
                selected = selected[:debug_max_slices_per_case]
            self.items.extend([(case_dir, s, False) for s in selected])

    def __len__(self):
        return len(self.items)

    def _augment(self, image, mask):
        """Apply augmentations."""
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        k = random.randint(0, 3)
        return torch.rot90(image, k, dims=[1, 2]), torch.rot90(mask, k, dims=[0, 1])

    def __getitem__(self, idx):
        case_dir, slice_idx, no_mask = self.items[idx] if len(self.items[idx]) == 3 else (self.items[idx][0], self.items[idx][1], False)

        sample = load_case(case_dir)
        mods = sample["modalities"]
        image = np.stack([mods["flair"], mods["t1"], mods["t1ce"], mods["t2"]], axis=0)[..., slice_idx, :, :]
        mask = sample["mask"]
        mask2d = np.zeros((image.shape[-2], image.shape[-1]), np.float32) if mask is None else label_map(mask[slice_idx], self.label_mode)

        image_t = torch.from_numpy(image.copy()).float()
        mask_t = torch.from_numpy(mask2d.copy())

        if image_t.shape[-2:] != (self.image_size, self.image_size):
            image_t = F.interpolate(image_t.unsqueeze(0), size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False).squeeze(0)
            mask_t = F.interpolate(mask_t[None, None].float(), size=(self.image_size, self.image_size),
                                  mode="nearest").squeeze()

        if self.training and self.augment:
            image_t, mask_t = self._augment(image_t, mask_t)

        if self.label_mode == "whole_tumor":
            mask_t = mask_t.unsqueeze(0).float()
        else:
            mask_t = mask_t.long()

        return image_t, mask_t, case_dir.name, slice_idx


def build_loaders(config: Dict, dataset_summary: Dict):
    """Create data loaders for train/val/test/official_val."""
    train_cases = dataset_summary["train_cases"]
    val_cases = dataset_summary["val_cases"]
    test_cases = dataset_summary["test_cases"]
    official_val_cases = dataset_summary["official_val_cases"]

    if config["USE_DEBUG_SUBSET"]:
        train_cases = train_cases[:config["DEBUG_NUM_CASES"]]
        val_cases = val_cases[:config["DEBUG_NUM_CASES"]]
        test_cases = test_cases[:config["DEBUG_NUM_CASES"]]
        official_val_cases = official_val_cases[:config["DEBUG_NUM_CASES"]]

    train_ds = BratsSliceDataset(
        train_cases, config["image_size"], config["label_mode"], True,
        config["debug_max_slices_per_case"] if config["USE_DEBUG_SUBSET"] else None,
        True, allow_missing_seg=False
    )
    val_ds = BratsSliceDataset(
        val_cases, config["image_size"], config["label_mode"], False,
        1 if config["USE_DEBUG_SUBSET"] else None, False, allow_missing_seg=False
    )
    test_ds = BratsSliceDataset(
        test_cases, config["image_size"], config["label_mode"], False,
        1 if config["USE_DEBUG_SUBSET"] else None, False, allow_missing_seg=False
    )
    official_val_ds = BratsSliceDataset(
        official_val_cases, config["image_size"], config["label_mode"], False,
        1 if config["USE_DEBUG_SUBSET"] else None, False, allow_missing_seg=True
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    official_val_loader = DataLoader(official_val_ds, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Official val batches (inference-only): {len(official_val_loader)}")

    if len(train_loader.dataset) == 0:
        print("\n[INFO] No training data available.")

    return train_loader, val_loader, test_loader, official_val_loader
