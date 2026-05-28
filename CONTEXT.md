# LETT-NeXt Context

LETT-NeXt means Lightweight Efficient Tumor Tracing with NeXt. This package reproduces the Ahus RECIST-to-3D submission path only. It is not a general experiment framework.

## Domain Terms

- Submitted package: the Docker inference package built from `artifacts/checkpoint.pt` and `predict.sh`.
- Canonical submitted run: the 3-GPU MedNeXt-v2 f32 run whose `best.pt` matches `artifacts/checkpoint.pt`.
- Fixed manifests: the train, validation, and auxiliary-evaluation manifests named in `configs/lett_next.yaml`.
- Split resampled cache: precomputed case cache at `data/prepared/cache/task2_resampled_1p0_1p0_2p4_component_aux_split_v1`.
- Dense RECIST prompt crop: crop sampling around RECIST endpoints with full prompt channels, multi-FOV crops, endpoint containment, and no RECIST augmentation.
- PanTS auxiliary head: the auxiliary anatomy supervision head enabled by `anatomy_aux_enable`.
- Metric-band retraining: retraining should reproduce the submitted recipe within a validation score band, not produce byte-identical weights.

## Reproducibility Contract

The submitted checkpoint is exact:

```text
sha256 artifacts/checkpoint.pt = ba7a4412aa4860c8e11cea860bd1462978763f0a95ace3a620ef21809d60ddb5
```

That file is byte-identical to the canonical run's `checkpoints/best.pt`. Docker inference is the exact reproducibility target. Training is the fixed 3-GPU recipe and is expected to land near the canonical validation score, not to recreate the same bytes.
