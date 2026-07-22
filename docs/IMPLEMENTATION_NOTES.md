# Implementation Notes — audited status

The implementation in this repository is a **lightweight research approximation (`ExUTransLite`)**, not a proven exact reconstruction of the published ExU-Trans model.

## Current components

- U-Net-style local encoder and decoder.
- Transformer token encoder with configurable Self-Explanatory Multi-Head Attention approximation.
- Discriminative Attribute Explainer approximation.
- Contextual spatial/channel attention.
- Bivariate Fusion Module approximation.
- Paper-inspired Trait-Guided Optimization objective: pixel loss + alignment + boundary regularization.

## Critical fidelity limits

- Refactored model: ~3.15M parameters.
- Paper Table 5: 50.3M parameters.
- Exact channel widths and several implementation details are not published.
- Repository split 70/15/15 is an implementation assumption; exact paper split IDs are not published.
- Default resize 128x128 is an implementation assumption; exact paper resize target is not published.
- `hd95_px` is grid/pixel distance. It must not be called millimetres without calibrated physical spacing.

## Evaluation policy

Validation/test slices are reconstructed into case volumes before metric computation. Actual outputs are written under `outputs/measured/`. Paper values are isolated under `outputs/reference/`.

For the complete defect history, evidence, and remaining blockers, see `../REPRODUCIBILITY_AUDIT.md`.
