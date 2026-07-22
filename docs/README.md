# ExU-Trans Brain Tumor Segmentation Reproduction

Complete implementation of the ExU-Trans brain tumor segmentation model on BraTS2020 dataset. This is a 2D slice-based approximation of the paper's 3D volumetric approach, optimized for practical execution on local hardware.

**Status**: ✅ **Complete and Ready for Delivery**

## Quick Start

1. **Setup**: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
2. **Run**: Open `notebooks/exu_trans_brats2020_reproduction.ipynb` and click "Run All"
3. **Results**: Metrics and figures auto-save to `outputs/`

See `CLIENT_README.md` for complete instructions.

## Target paper

- Title: ExU-Trans: a self-explanatory transformer with U-Net based hybrid model for brain tumor segmentation using MR imaging.

## Dataset source

- Dataset: BraTS2020 Training + Validation.
- The dataset was obtained externally and is stored locally in this repository tree under `dataset/`.
- Extracted training root: `dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/`
- Extracted validation root: `dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/`

## Local data placement

- Recommended local root: `dataset/`
- Training data: `dataset/BraTS2020_TrainingData/`
- Validation data: `dataset/BraTS2020_ValidationData/`
- The extracted subject folders, CSV metadata, and imaging files remain local only.

## How to download / prepare data

1. Register and download BraTS2020 from the official source or Kaggle if you have access.
2. Place the downloaded BraTS files under the local `dataset/` directory.
3. Keep the original directory structure so notebooks and scripts can reference the data.

## Included Kaggle notebooks

- `codes/attention-unet-v2.ipynb`
- `codes/brats2020-mapvit-100epochs.ipynb`
- `codes/u-net-transformer.ipynb`

## Repository Structure

```
├── notebooks/
│   └── exu_trans_brats2020_reproduction.ipynb  <- MAIN NOTEBOOK (run this)
├── dataset/                                      <- BraTS2020 (download separately)
├── outputs/                                      <- Auto-generated results
│   ├── metrics/metrics_validation.csv
│   ├── figures/figure_4_metrics_comparison.png
│   └── figures/figure_segmentation_results.png
├── CLIENT_README.md                             <- Client delivery guide
├── IMPLEMENTATION_NOTES.md                      <- Technical details
├── COMPARISON_ANALYSIS.md                       <- Detailed paper comparison
├── GET_STARTED.md                               <- Setup instructions
├── requirements.txt                             <- Dependencies
└── article/s40747-026-02279-3.pdf              <- Original paper
```

## Key Results

- **Dice**: 76.36% ± 4.77% (2D approximation vs 90.6% in paper's 3D)
- **IoU**: 68.70% ± 5.73%
- **All 6 metrics computed**: Dice, IoU, Precision, Recall, F1, Hausdorff Distance
- **Publication-quality figures**: 300 DPI PNG exports
- **CSV metrics**: Per-case and aggregate statistics

## 2D vs 3D Trade-off

This implementation uses **2D slice-based processing** instead of 3D volumetric:
- ✅ Runs on local hardware (CPU/GPU)
- ✅ Executes in <10 minutes
- ✅ All architecture components present
- ⚠️ Expected 10-20% metric gap from paper (actual: 14.24%)

See `IMPLEMENTATION_NOTES.md` for architectural details and metric explanations.

## Important Notes

- Dataset files are intentionally ignored and must not be pushed to GitHub.
- This repository contains only source code, notebooks, documentation, and lightweight metadata.
- Download BraTS2020 separately from: https://www.med.upenn.edu/cbica/brats2020/
