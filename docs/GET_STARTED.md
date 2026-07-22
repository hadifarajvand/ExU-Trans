# Phase 3: Get Started - Run and Generate Outputs

## Quick Start (2 minutes)

### 1. Enable Output Export

Open the notebook:
```bash
jupyter notebook notebooks/exu_trans_brats2020_reproduction.ipynb
```

In **cell-2** (Configuration section), change:
```python
CONFIG["SAVE_OUTPUTS"] = False
```

To:
```python
CONFIG["SAVE_OUTPUTS"] = True
```

### 2. Run All Cells

Click: **Cell** → **Run All** (or press `Ctrl+A` then `Shift+Enter`)

Expected time: ~1-2 minutes with GPU, ~5 minutes on CPU (debug mode)

### 3. Check Outputs

All results automatically generated in `outputs/` folder:

```
outputs/
├── metrics/
│   └── metrics_validation.csv          ← Per-case metrics table
├── figures/
│   ├── figure_4_metrics_comparison.png ← Metrics visualization
│   └── figure_segmentation_results.png ← Segmentation grids
└── attention_maps/
    └── attention_maps_*.png             ← Attention visualizations
```

---

## What Gets Generated

### **1. Metrics CSV** (`metrics_validation.csv`)

Per-case metrics for each test case:

```
case_id,dice,iou,hd95,precision,recall,f1
BraTS20_Training_001_slice_85,0.823,0.745,4.23,0.851,0.796,0.823
BraTS20_Training_002_slice_92,0.792,0.712,5.12,0.823,0.765,0.794
...
```

Plus aggregate statistics:
```
Summary statistics printed to console:
  Mean Dice:     0.803 ± 0.045
  Mean IoU:      0.721 ± 0.052
  Mean Precision: 0.815 ± 0.038
  Mean Recall:    0.782 ± 0.048
  Mean F1:        0.798 ± 0.043
  Mean HD95:      4.89 ± 1.23 mm
```

### **2. Figure 4: Metrics Comparison**

2×3 grid of histograms showing distribution of:
- Dice Similarity Coefficient
- Intersection over Union
- Hausdorff Distance
- Precision
- Recall
- F1 Score

**File:** `outputs/figures/figure_4_metrics_comparison.png` (300 DPI)

### **3. Segmentation Results**

Multi-case visualization with 5 columns:
1. Input FLAIR image
2. Ground truth mask
3. Model prediction
4. Overlay (GT + prediction)
5. Difference heatmap

**File:** `outputs/figures/figure_segmentation_results.png` (300 DPI)

### **4. Attention Maps** (Interpretability)

Per-case attention visualizations (2×3 grid):
- FLAIR input
- Ground truth mask
- DAE attribute map (from Discriminative Attribute Explainer)
- SE-MHA attention map (from Self-Explanatory Multi-Head Attention)
- Spatial attention weights
- Channel attention weights

**Files:** `outputs/attention_maps/attention_maps_*.png`

---

## Understanding the Results

### **Expected Metrics (2D Slice-Based)**

```
Your Results:          Paper (3D):        Reason for Difference:
Dice: 70-80%           Dice: 90.2%        2D slices vs. 3D volumes
IoU: 60-70%            IoU: 84.1%         Per-slice context only
HD95: 5-8mm            HD95: 3.0mm        No volumetric information
```

**This is NORMAL.** The notebook is a 2D approximation, not an exact 3D reproduction. The workflow, formulas, and components match the paper exactly - only the approach differs.

### **What DOES Match the Paper**

✓ All 8 workflow steps  
✓ All loss functions (TGOF with 3 components)  
✓ All metric definitions  
✓ Model architecture (SE-MHA, DAE, BFM, CSA)  
✓ Preprocessing pipeline  
✓ Output format  

### **Loss Breakdown** (Printed During Training)

```
Training with TGOF (Trait-Guided Optimization Function):
  Total Loss: 0.6234
    - BCE (Pixel-level): 0.3456
    - Dice Loss: 0.2123
    - Alignment Loss (DAE): 0.0456
    - Boundary Loss (Edge): 0.0199
```

The TGOF loss has 3 components matching the paper exactly:
1. L_pixel = L_BCE + L_Dice (supervised pixel-level loss)
2. L_align = MSE(A_trait, σ(F_trait)) (trait-guided alignment)
3. L_boundary = MSE(∇S_pred, ∇S_true) (boundary-aware regularization)

---

## Using the Results

### **For a Research Report**

1. Copy CSV metrics:
   ```bash
   cp outputs/metrics/metrics_validation.csv my_report/
   ```

2. Add figures to document:
   ```markdown
   ![Metrics Comparison](outputs/figures/figure_4_metrics_comparison.png)
   ![Segmentation Results](outputs/figures/figure_segmentation_results.png)
   ```

3. Add mean metrics from console output

### **For Further Analysis**

Load metrics into pandas:
```python
import pandas as pd
df = pd.read_csv("outputs/metrics/metrics_validation.csv")

# Summary statistics
print(df[['dice', 'iou', 'f1']].describe())

# Visualizations
df.boxplot(column=['dice', 'precision', 'recall'])
```

### **For a Paper**

1. Use Figure 4 as-is (300 DPI, publication-ready)
2. Use segmentation results grid
3. Include attention maps to show interpretability
4. Cite that this is a 2D approximation of ExU-Trans

---

## Configuration Options

### **To Run with More/Fewer Cases**

```python
# Use 10% of data instead of debug 2 cases
CONFIG["USE_DEBUG_SUBSET"] = True
CONFIG["DEBUG_NUM_CASES"] = 37

# Use all data (slow)
CONFIG["USE_DEBUG_SUBSET"] = False
```

### **To Run with Full Training**

```python
CONFIG["USE_DEBUG_SUBSET"] = False
CONFIG["epochs"] = 10
CONFIG["SAVE_OUTPUTS"] = True
```

### **To Save Model Checkpoints**

```python
CONFIG["SAVE_CHECKPOINTS"] = True
```

(Checkpoints saved to `checkpoints/` directory)

---

## Troubleshooting

### **"No module named 'X'" error**

Install dependencies:
```bash
pip install -r requirements.txt
```

### **CUDA out of memory**

Reduce batch size in config:
```python
CONFIG["batch_size"] = 1
```

### **Very slow execution**

You're running on CPU. GPU is ~10x faster:
- Check: `nvidia-smi`
- If no GPU available, reduce `DEBUG_NUM_CASES` for faster testing

### **Outputs not generated**

Make sure:
1. `CONFIG["SAVE_OUTPUTS"] = True`
2. `outputs/` directory exists (created automatically)
3. Check console for "Exporting results..." message

---

## File Reference

| File | Purpose |
|------|---------|
| `OUTPUTS_DOCUMENTATION.md` | What each output means |
| `PHASE3_COMPLETION_REPORT.md` | Implementation details |
| `PHASE3_FINAL_SUMMARY.md` | Complete summary |
| `PHASE3_PLAN_PAPER_ALIGNMENT.md` | Implementation plan |
| `QUICK_START.md` | Basic setup instructions |

---

## Next Steps

After running and collecting outputs:

1. **Review metrics** - Check if dice/iou values make sense
2. **Examine figures** - Qualitatively assess segmentations
3. **Analyze attention maps** - See what model focuses on
4. **Further experiment** - Try different configs, longer training
5. **Share results** - Use figures and metrics for reports

---

## Summary

✓ Set `CONFIG["SAVE_OUTPUTS"] = True`  
✓ Run notebook (Cell → Run All)  
✓ Check `outputs/` folder for results  
✓ Use metrics CSV and figures for reports  

**That's it!** The notebook handles everything else automatically.

---

For detailed information about each output, see `OUTPUTS_DOCUMENTATION.md`.
