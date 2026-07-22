# ExU-Trans Code / Dataset Mapping

## Local Paths
- Paper PDF: `D:/GitHub/aysan/class-projects/1/article/s40747-026-02279-3.pdf`
- Training dataset root: `D:/GitHub/aysan/class-projects/1/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
- Validation dataset root: `D:/GitHub/aysan/class-projects/1/dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData`

## Detected Dataset Structure
- Training folder exists: yes
- Validation folder exists: yes
- Training case count: 369 (all labeled with segmentation masks)
- Official validation case count: 125 (unlabeled; no segmentation masks)
- Training case naming pattern: `BraTS20_Training_###`
- Official validation case naming pattern: `BraTS20_Validation_###`
- Training files per case: `*_flair.nii`, `*_t1.nii`, `*_t1ce.nii`, `*_t2.nii`, `*_seg.nii`
- Official validation files per case: `*_flair.nii`, `*_t1.nii`, `*_t1ce.nii`, `*_t2.nii` (no `*_seg.nii`)

**Critical distinction:** The official validation folder contains no segmentation masks. Therefore, it is unsuitable for supervised metric evaluation (Dice, IoU, HD95, etc.). The notebook uses an **internal train/val/test split from the 369 labeled training cases** and treats the 125 official validation cases as **inference-only** (blind test) data.

## Detected Modalities
- FLAIR
- T1
- T1ce / T1Gd
- T2
- Segmentation mask in training only

## Downloaded Kaggle Notebooks / Code
1. `D:/GitHub/aysan/class-projects/1/codes/u-net-transformer.ipynb`
2. `D:/GitHub/aysan/class-projects/1/codes/brats2020-mapvit-100epochs.ipynb`
3. `D:/GitHub/aysan/class-projects/1/codes/attention-unet-v2.ipynb`

## Notebook-Level Mapping

### `u-net-transformer.ipynb`
- Model architecture: TensorFlow/Keras U-Net + transformer encoder hybrid.
- Dataset loader: NIfTI-based loader over BraTS case folders.
- Preprocessing logic: slice extraction, resizing, MinMax scaling, paired modality input.
- Training loop: Keras `fit()` with callbacks.
- Validation / inference: slice-wise prediction helpers and case visualization.
- Metrics: Dice, precision, sensitivity, specificity, MeanIoU.
- Reusable for ExU-Trans reproduction: paired modality loading, per-slice segmentation workflow, metric structure, and a compact transformer-U-Net reference.

### `brats2020-mapvit-100epochs.ipynb`
- Model architecture: MobileNetV2 encoder + dense pyramid + transformer + attention-gated U-Net decoder.
- Dataset loader: HDF5 slice loader.
- Preprocessing logic: Albumentations and slice normalization, whole-tumor mask collapse.
- Training loop: PyTorch training with 5-fold cross-validation framing.
- Validation / inference: test evaluation and sample evaluation utilities.
- Metrics: mDice, mIoU, bootstrap confidence intervals, Grad-CAM explainability hooks.
- Reusable for ExU-Trans reproduction: PyTorch pipeline structure, augmentation patterns, transformer bottleneck ideas, and explainability visualization patterns.

### `attention-unet-v2.ipynb`
- Model architecture: TensorFlow attention U-Net.
- Dataset loader: HDF5 slice loader.
- Preprocessing logic: generator-based image/mask pairing.
- Training loop: Keras `fit()` pipeline.
- Validation / inference: random-slice prediction visualization.
- Metrics: Dice and combined loss functions.
- Reusable for ExU-Trans reproduction: attention-gated decoder ideas, compact generator structure, and qualitative comparison layout.

## ExU-Trans Component Mapping

### Directly Implemented in the Consolidated Notebook
- BraTS2020 case discovery.
- Multi-modal loading for FLAIR, T1, T1ce/T1Gd, and T2.
- Non-background z-score normalization.
- Binary whole-tumor labels and optional BraTS subregion support.
- Dual-encoder approximation with U-Net + ViT branches.
- Dice, IoU, precision, recall, F1, and HD95-style metrics.
- Tiny debug training and inference flow.

### Approximated Because Exact Details Were Not Fully Available Locally
- SE-MHA: implemented as a practical multi-head attention block with a learned explanatory bias term.
- DAE: implemented as a lightweight attribute scoring head and attribute map projection.
- Contextual self-attention: implemented as a CBAM-like spatial/channel refinement block.
- Bivariate fusion: implemented as attention-based fusion of aligned local and global features.
- Attribute-guided loss: implemented as an auxiliary BCE alignment loss between the attribute map and tumor mask.
- Full 3D training: notebook is **slice-based (2D)** for local practicality — **NOT a true 3D reproduction**.

### Implementation Scope: 2D Slice-Based Approximation
The current implementation uses 2D slices extracted from 3D volumes:
- Each case is processed as independent 2D slices (no temporal/volumetric context).
- Model input: (B, C=4, H=128, W=128) per slice (not 3D volume).
- Model output: (B, 1, H=128, W=128) per slice.
- Metrics (Dice, IoU, HD95) computed per-slice, not per-volume.

**Result:** This is an approximate, practical reproduction suitable for local hardware and educational purposes, **not an exact match to the full 3D ExU-Trans model**.

## Dataset Split Strategy (Updated)

**Labeled Training Cases (369 total):**
- Internal train split: 70% (~259 cases) — used for weight updates
- Internal val split: 15% (~55 cases) — used for metric evaluation during training
- Internal test split: 15% (~55 cases) — reserved for final metric reporting

**Official Validation Cases (125 total):**
- Treated as **unlabeled inference-only** test set (no ground truth masks)
- Used only for blind predictions and qualitative visualization
- NO metric computation on official validation cases
- Can be used post-training to assess generalization to unseen data without labels

## Assumptions
- The 369 labeled training cases are the sole source of ground truth for supervised training and metric evaluation.
- The 125 official validation cases are held out for inference-only assessment.
- The paper descriptions are sufficient to build a faithful approximation, but not a byte-for-byte reproduction.
- Whole-tumor binary segmentation is the default reproduction target.
- Subregion support is kept as an optional path in the notebook.

## Blockers / Missing Information
- The exact official ExU-Trans source code is not locally available.
- The local PDF did not yield full text through the installed shell PDF parsers, so the paper methodology was confirmed through the Springer article page and PDF metadata.
- The current shell Python environment does not have the scientific stack preinstalled, so the notebook includes a bootstrap installation comment rather than assuming local packages are present.
- **No 3D volume processing:** The notebook uses 2D slices for practical execution on local hardware. A true 3D reproduction would require volumetric convolutions and transformers, which is beyond the scope of this reproduction.

## Validation & Testing Strategy
- **Labeled validation set (15% of training, ~55 cases):** Used during development to monitor model performance and compute metrics.
- **Labeled test set (15% of training, ~55 cases):** Reserved for final metric reporting after hyperparameter tuning.
- **Official validation set (125 cases, unlabeled):** Blind inference test — used only for predictions and visualizations, not metrics.
- **Reproduction targets:** Dice, IoU, Precision, Recall, F1 (all computed on labeled splits only).
- **Approximate but not exact:** Due to the slice-based approach, reported metrics will not match the paper's full 3D results (paper reports 0.962 Dice for whole tumor on 3D volumes).

## Paper-Method Summary
- Input: multimodal MRI.
- Architecture: dual encoder, ViT + U-Net.
- Explainability: SE-MHA and DAE.
- Fusion: spatial + channel attention via BFM.
- Loss: attribute-guided optimization.
- Metrics: Dice / DSC, IoU / Jaccard, HD / HD95, Precision, Recall, F1.
- Reported BraTS2020 results: mean Dice 0.962 (WT), 0.941 (ET), 0.944 (TC); the paper also reports improved Jaccard/HD versus baselines and cross-dataset generalization.
