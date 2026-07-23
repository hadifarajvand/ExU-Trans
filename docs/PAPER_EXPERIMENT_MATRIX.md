# PAPER_EXPERIMENT_MATRIX

Inventory of paper claims versus repository support. Status is current codebase status, not claimed reproduction.

| Paper item | Type | Paper experiment | Required data | Required model/run | Metric/output | Current repo support | Reproducible? | Execution status | Output path | Notes/limitations |
|---|---|---|---|---|---|---|---|---|---|---|
| Comparative benchmark table | Table | Compare ExU-Trans vs 5 baselines | BraTS2020 test set | Full inference on published or reimplemented baselines | DSC, IoU, HD, Precision, Recall, F1 | Partial | No full baseline suite | Not executed here | `reference/paper_targets.json`, `outputs/reference/` | Baseline code and matching evaluation protocol not present |
| Significance table | Table | t-test / Wilcoxon vs baselines | Per-case paired scores | Paired statistical testing | p-values | Partial | No | Not executed here | `reference/paper_targets.json` | Aggregate means alone are insufficient |
| Dataset-wise table | Table | BraTS2020 and BraTS2021 scores | Both datasets | Full evaluation on both datasets | DSC, IoU, HD, Precision, Recall, F1 | Partial | No | Not executed here | `reference/paper_targets.json` | BraTS2021 path not validated |
| Noise robustness | Table | Add label/noise perturbation | BraTS2020 with injected noise | Controlled noisy runs | DSC, IoU, HD, Precision, Recall, F1 | No | No | Not executed here | `reference/paper_targets.json` | No noisy-eval runner implemented |
| Efficiency table | Table | Runtime, params, GPU memory, FLOPs | Model graph + profiling hardware | Profiling run | inference_s, params, memory, FLOPs | Partial | No | Not executed here | `reference/paper_targets.json` | FLOPs/memory not instrumented |
| Generalization table | Table | Cross-dataset evaluation | External datasets | Inference on ACRIN-6677 / T1 / T2 | DSC, IoU, HD, Precision, Recall, F1 | No | No | Not executed here | `reference/paper_targets.json` | No data loaders for these datasets |
| Tumor size table | Table | Small vs large tumors | Tumor-size stratification metadata | Stratified evaluation | DSC, IoU, HD, Precision, Recall, F1 | No | No | Not executed here | `reference/paper_targets.json` | No size stratification metadata loader |
| Ablation table | Table | Full model vs modules removed | BraTS2020 | Architectural ablations | DSC, Precision, Recall | Partial | No | Not executed here | `reference/paper_targets.json` | Core modules exist only as lightweight approximation |
| Loss ablation table | Table | Trait-guided optimization on/off | BraTS2020 | Loss ablation runs | DSC, Precision, Recall | Partial | No | Not executed here | `reference/paper_targets.json` | Loss composition present, but no paper-equivalent protocol proof |
| Hyperparameter table | Table | LR/layers/fusion sweep | BraTS2020 | Grid-search or controlled sweeps | DSC across settings | Partial | No | Not executed here | `reference/paper_targets.json` | Current config aligns to paper targets, but sweep outputs missing |
| Segmentation qualitative figures | Figure | Predicted masks vs GT | Sample cases | Eval on held-out cases | Overlay images | Partial | Yes, if dataset exists | Not executed here | `outputs/measured/figures/` | Requires real dataset and export run |
| Attention maps | Figure | Self-explanation visualizations | Sample cases | Forward pass + export | Attention / attribute maps | Partial | Yes, if dataset exists | Not executed here | `outputs/measured/attention_maps/` | Current model exposes attention tensors |
| Training curves | Figure | Loss / metric curves | Training logs | Full training run | Line plots | Partial | Yes, if logs exist | Not executed here | `outputs/measured/figures/` | Depends on training logs being saved |
| Paper target registry | Reference | Publish paper numbers only | None | No model run | JSON/CSV reference tables | Yes | Yes | Exported, not measured | `reference/paper_targets.json`, `outputs/reference/` | Reference only; not research output |
| Measured metrics CSV | Measured | Case-level scores from actual run | BraTS2020 cases | Real training/eval pipeline | Per-case CSV + aggregates | Partial | No run completed here | Not executed here | `outputs/measured/metrics/` | Missing dataset/deps in this environment |
| Comparison report | Comparison | Measured vs paper delta | Measured CSV + reference JSON | Compare script | Delta report | Yes | Yes | Not executed here | `outputs/comparison/` | Works only after measured CSV exists |
| Smoke test | Smoke | Sanity check only | Synthetic data | Synthetic forward/backward step | Synthetic loss/dice/iou | Yes | Yes | Available | `outputs/smoke/` | Not a reproduction result |

## Bottom line

- Full-dataset paper reproduction: not proven in this workspace.
- Reference export: implemented.
- Measured end-to-end run: blocked until BraTS2020 data and runtime deps exist.
- Comparison tooling: implemented.
- Additional figures/tables: only partial support from current code.
