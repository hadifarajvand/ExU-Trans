# ExU-Trans Reproducibility Audit

Date: 2026-07-22

## Verdict

The pre-audit repository was runnable as a plotting/export demo but was **not a defensible paper reproduction**. The largest failure was provenance: missing measured metrics triggered random synthetic generation, while several paper-style tables were hard-coded and then described as verified/matching outputs.

## Critical findings

### 1. Silent synthetic fallback

The old `scripts/export_paper_results.py` created random metrics with `np.random.normal(...)` whenever `outputs/metrics/metrics_validation_full.csv` was absent. That metrics file is not present in the repository. Therefore the default quick-export path did not prove a dataset simulation.

**Resolution:** removed the random fallback. Published values now export only from `reference/paper_targets.json` under `outputs/reference` with `REFERENCE_ONLY` labeling.

### 2. Hard-coded results were presented as simulation results

Old tables for baselines, cross-dataset generalization, noise robustness, ablation, and efficiency were hard-coded. Several values did not match the 2026 paper.

**Resolution:** paper values are separated from measured outputs. Measured results are written only under `outputs/measured`.

### 3. Architecture fidelity was overstated

The old README called the implementation a complete reproduction. A direct smoke audit showed approximately 2.72M parameters and two Transformer blocks. The paper reports 50.3M parameters and Table 13 identifies four Transformer layers as optimal.

**Resolution:** model is explicitly documented as `ExUTransLite`, defaults to four Transformer blocks, and no exact-architecture claim is made. The refactored model smoke test is approximately 3.15M parameters, still far from the paper's 50.3M; that mismatch is disclosed rather than hidden.

### 4. Paper component names were wrong

Old documentation incorrectly expanded the module names. The paper defines:

- SE-MHA: Self-Explanatory Multi-Head Attention
- DAE: Discriminative Attribute Explainer
- BFM: Bivariate Fusion Module

**Resolution:** corrected terminology and approximation warnings.

### 5. Loss bug

For binary segmentation the old loss added the same alignment term twice through `attr_weight * align + align_weight * align`. It also applied `sigmoid` to a binary target mask before MSE, mapping target 0/1 to approximately 0.5/0.731.

**Resolution:** the objective is now explicitly `pixel + alignment + boundary`, each included once; the alignment target remains binary.

### 6. HD95 unit error

The old code computed `distance_transform_edt` on resized arrays without physical spacing, then documentation/tables labeled the result `mm`.

**Resolution:** uncalibrated distance is now named `hd95_px`. The paper comparison script does not compare `hd95_px` against paper millimetres.

### 7. Evaluation granularity was biased

The old dataset selected tumor-containing slices, and evaluation averaged per-slice scores. This is not equivalent to case/volume-level evaluation and can bias results.

**Resolution:** validation/test enumerate all slices, reconstruct predictions by case, and compute metrics at the case-volume level. Training may still sample tumor-containing slices as an implementation choice.

### 8. Best checkpoint was never actually used for final test

The old main loop tracked `best_val_dice` but did not restore the corresponding weights and did not perform a clear final test evaluation.

**Resolution:** best state is copied on validation improvement, restored after training, saved, and then used for final validation/test export.

### 9. Non-portable paths

The old configuration hard-coded a developer-specific Windows path.

**Resolution:** dataset paths now use project-relative defaults and environment variables.

## Paper inconsistencies that prevent a single target score

The paper itself is internally inconsistent:

- Table 1: ExU-Trans DSC = 90.2%.
- Table 3: BraTS2020 DSC = 90.6%.
- Tables 11-13: full/optimized model DSC = 92.0%.
- Narrative text: WT/ET/TC Dice = 0.962/0.941/0.944.

Therefore this repository preserves table-specific targets. A comparison must state which target is used.

## Local execution evidence

### Pre-refactor model smoke audit

- Input: `(2, 4, 128, 128)`
- Output: `(2, 1, 128, 128)`
- Transformer blocks: 2
- Parameter count: 2,719,948 (~2.72M)
- Status: forward pass PASS

### Refactored smoke audit

- Input: `(1, 4, 128, 128)`
- Output: `(1, 1, 128, 128)`
- Transformer blocks: 4
- Parameter count: 3,149,516 (~3.15M)
- Forward/loss/backward/optimizer: PASS
- Mode: `synthetic_smoke_only`

Synthetic Dice/IoU from smoke are intentionally not interpreted as research performance.

### Old advertised means vs paper Table 1

Using the old README's advertised aggregate means as a diagnostic comparison:

| Metric | Old advertised | Paper Table 1 | Gap |
|---|---:|---:|---:|
| DSC | 80.61% | 90.2% | -9.59 pp |
| IoU | 67.57% | 84.1% | -16.53 pp |
| Precision | 82.56% | 89.6% | -7.04 pp |
| Recall | 79.30% | 87.7% | -8.40 pp |
| F1 | 80.91% | 88.6% | -7.69 pp |

HD95 is excluded because the old implementation's value was grid/pixel distance, not calibrated millimetres.

## Remaining hard blocker for a full numerical reproduction

The dataset/code snippet described for this audit was not available in the active upload workspace. The GitHub repository contained code but no committed real BraTS data or measured metric files. Therefore a truthful full-dataset rerun could not be performed in this audit environment.

This is not solved by copying paper numbers into outputs. Full numerical reproduction requires the actual authorized dataset (or the user's intended snippet for execution testing) and a measured run.

## Scientific status after refactor

- Code-path smoke execution: PASS
- Fake/silent result generation: REMOVED
- Paper target registry: PASS
- Explicit measured-vs-reference comparator: PASS
- Portable configuration: PASS
- Case-level evaluation path: IMPLEMENTED
- Best-checkpoint test path: IMPLEMENTED
- Exact paper architecture reproduction: NOT PROVEN
- Full BraTS rerun in this environment: BLOCKED BY MISSING DATA
- Exact paper-number match: NOT CLAIMED
