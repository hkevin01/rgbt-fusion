# RGB-T Fusion Architecture

The framework exposes one configurable model entrypoint with four fusion strategies:

- Early fusion: Concatenate RGB and thermal at input and project to 3 channels.
- Mid fusion: Encode each modality, fuse features with gated attention.
- Late fusion: Independent modality logits and learnable score weighting.
- Research fusion: MMTM-style cross-modal channel recalibration and fusion.

Two tasks are included:

- Classification: global pooled fused feature to class logits.
- Semantic segmentation: fused feature decoder to per-pixel logits.
