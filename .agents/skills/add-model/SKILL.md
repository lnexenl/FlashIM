---
name: add-model
description: Use when adding a model, sparse extractor, pipeline or a new weights variant to FlashIM (a native PyTorch port of an upstream image-matching model). Covers layout, weights pinning, presets, parity test, calibration, licenses, docs and the accuracy gate.
---

# add-model

A model is a native PyTorch port that is checked against the upstream code (the "oracle"). Follow the closest existing
model of the same kind (dense: `roma`, `romav2`, `ufm`; semi-dense: `loftr`, `eloftr`, `xoftr`; sparse extractor:
`superpoint`; matcher: `lightglue`, `superglue`; pipeline: `superpoint-superglue`, `omniglue`) instead of designing anew.

## Layout (`flashim/models/<m>/`)

`model.py` (pure `nn.Module`, calls `ops.*` only, never a backend), `config.py`, `weights.py`, `quant.py` (presets and
their accuracy evidence), `spec.py` (`ModelSpec`), registered in `flashim/registry.py` `SPECS`.

- The class is an `ImageMatcher` (implement `_match`) or a `FeatureMatcher` (`extract` + `match_features`) and returns
  `flashim.Matches`. Per-call defaults come from the config's `inference` section (`match_defaults`). Anything
  model-specific is a config key, never a new `flashim.load` keyword or a per-model `load_*`.
- Build from `flashim.layers` where the pattern fits (`TransformerBlock`, `SeparableConvBlock`, ...) so quantization and
  fusion passes apply by layer type. Keep upstream parameter names so the checkpoint loads with `strict=True`.
- Behaviour is a named `Pass` in `spec.py` (stage `pre_quant` / `post_quant`, `requires=` a platform predicate, `exact=`),
  never a flag flipped by hand in a loader. Configs kept for parity or experiments are internal overlays
  (`ORACLE_OVERLAYS` in RoMaV2), not presets.
- Presets are config overlays (`extends` allowed) named only `fp32` / `bf16` / `int8` / `fp8`. A preset the model lacks
  builds the nearest one (`build.NEAREST`). A pipeline whose component's fp8 is strictly worse than its int8 omits `fp8`
  from its presets (`_PIPELINE_PRESETS` in `superglue/spec.py`, `omniglue/spec.py`).
- Sparse extractor: subclass `SparseExtractor` (`models/sparse.py`; implement `detect`, `batchnorm_pairs`), build the spec
  with `extractor_spec(quantized=..., protected=...)`. Convolutions are quantized as im2col + GEMM
  (`quant/methods/conv2d.py`, groups == 1 only); on the small nets (< 128 channels) that measured slower than BF16.

## Weights and licenses

- `weights.py` pins every release as `Checkpoint(filename, url, sha256)` and loads it through `find_weights`
  (`flashim/models/weights_util.py`: `checkpoint` config key, env var, torch hub folder, else download). Pin URLs to a
  commit / release, not a branch; `sha256sum` the file you downloaded. `tests/unit/test_architecture.py` fails until the
  module is in `WEIGHT_MODULES` of `tests/unit/test_weights.py`, and `test_every_model_pins_its_release` there also counts
  the distinct checkpoints (`assert len(found) == N`): bump N. If the only source is a drive link or a form, ask the user
  where it may be mirrored instead of pinning an unstable URL.
- Put the code and weights licenses in `LICENSES`, and add the model to the README (supported-model list, license list),
  `docs/models.md` (a row; a page under `docs/models/` is optional) and `THIRD_PARTY_NOTICES.md`. Flag non-commercial
  weights (SuperPoint, SuperGlue, RIPE, MASt3R / DUSt3R / UFM, DINOv3).
- A model that ships only some presets (e.g. `fp32` and `bf16`) still has a short `quant.py` saying which presets exist
  and why; the others are not registered and build the nearest one.
- A matcher whose inputs are not keypoints + descriptors (lines, regions) does not fit `SparseFeatures` /
  `FeatureMatcher.match_features`: ask before extending those shared types.

## Tests

- CPU unit tests (`tests/unit/test_<m>_cpu.py`) and a parity test against the oracle (`tests/parity/<m>/test_parity.py`,
  fp32). Vendor only the upstream files the test imports into `tests/oracles/<m>/`, each with a license header and the
  upstream LICENSE next to it, and list it in `tests/oracles/README.md`. If the upstream license forbids redistribution,
  fetch it on demand (`tests/parity/_fetch.py`, SuperGlue).
- Parity pitfalls: put upstream modules in `.eval()`; disable `cudnn.allow_tf32` (BN folding changes TF32 rounding);
  `torch.cdist` on pixel coordinates needs `compute_mode="donot_use_mm_for_euclid_dist"`; compare keypoints as sets, not by
  index (a few threshold-borderline points flip and shift every later index).

## Quantization and the accuracy gate

1. Presets that build are not presets that are accurate: use the `quantization-accuracy` skill when a quantized preset
   loses accuracy, and calibrate (`scripts/calibrate_extractor.py --model m --format fp8 --format int8`, LightGlue
   pairings `scripts/calibrate_lightglue.py --extractor m`, or the model's own `scripts/calibrate_<m>.py`).
2. Add the model to `benchmarks/accuracy/b_models_protocol.py` (one protocol for all presets of a model, native
   resolution) and run `b_models_matrix.sh` + `b_models_report.py` (see the `verify-change` skill). Report what was not
   verified (other GPU archs are compile-checked only).
3. Kernels for the model: `add-op` and `optimize-kernel`.
