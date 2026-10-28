# Configuring a run

Every model is built by one generic builder from a `RunConfig` (`flashim/config.py`). A preset is a named partial
config; files, dicts, dotted overrides and `load`'s keyword arguments layer on top.

## Loading

```python
model = flashim.load("romav2", "fp8", engine="compile-full", overrides={"arch.setting": "mega1500"})
model = flashim.load("raco-aliked-lightglue", "int8", config="run.toml")   # .toml / .json (.yaml with PyYAML)
model = flashim.load("aliked", "bf16", device="cpu", overrides=["variant=aliked-t16", "optimize.passes+=channels_last"])
```

`flashim.load(name, preset, *, device, engine, strict, calibration, config, overrides)` is the one way to get a model.
`device`, `engine` and `strict` apply to every component of a pipeline; `calibration` (a table path or a loaded table)
to single models only. Everything model-specific (`arch.setting`, `variant`, `checkpoint`, `optimize.attention`,
`engine.mode`, ...) is a config key.

## Presets

Every model's presets are among `fp32` (checkpoint precision, the upstream numerics), `bf16`, `int8` (static INT8) and
`fp8` (static MXFP8). A model without the one asked for builds the nearest it has, in the order `fp8` -> `int8` -> `bf16`
-> `fp32` (`fp32` -> `bf16`): RoMaV2 has no `fp32`, RDD no `int8`, ALIKED and RaCo neither `int8` nor `fp8`, SIFT and ORB
only `fp32`. A pipeline's preset is each component's, and each builds its own nearest one: `aliked-lightglue` `int8` runs
ALIKED in `bf16` and LightGlue in `int8`. On a GPU without FP8 `fp8` is replaced the same way ([platforms.md](platforms.md)).
What was replaced is logged and listed in `model.build_report.fallbacks`; `optimize.strict=true` raises instead.

## Matching

Every model that matches an image pair answers the same call:

```python
m = model.match("a.jpg", "b.jpg", resize=1024, max_matches=2048, threshold=None)   # flashim.Matches
m.keypoints_A, m.keypoints_B     # [N, 2] (x, y) pixels of each original image (integer = pixel center)
m.confidence                     # [N], the model's own score
m.indices_A, m.features_A        # detector/descriptor models: indices into each image's SparseFeatures
m.dense                          # dense models: the full DenseMatchResult (warp, overlap, precision)
```

| option | meaning |
|---|---|
| `resize` | long side of the working resolution. RoMaV2 rejects it: its resolutions are `arch.setting`'s |
| `max_matches` | at most this many matches: keypoints per image (detector/descriptor models), the most confident matches (LoFTR), balanced samples of the dense warp (RoMaV2, default 5000) |
| `threshold` | the model's match threshold: mutual-NN cosine, dual-softmax probability, LightGlue's filter, LoFTR's coarse confidence. RoMaV2 has none |

Options left `None` take the run config's `inference` section (`inference.max_matches=1024`), then the model's default.
Detector/descriptor models also have `extract(image, resize=, top_k=)` and `match_features(a, b, threshold)`; RoMaV2 has
`match_dense(a, b)` and `sample(dense, n)`. Keypoints of every model are in pixel-center coordinates of the original
image (integer = pixel center), RaCo's included.

## Sections

| section | what it holds |
|---|---|
| top level | `model`, `variant`, `checkpoint`, `device` |
| `arch` | model-specific architecture fields. They override the checkpoint's config and change the graph |
| `precision` | compute dtype, quantization config, calibration table, per-module precision overrides |
| `optimize` | named passes (model-specific fusions), attention implementation, op overrides, `strict`, pass options |
| `engine` | `kind` (`eager`, `compile`, ...), compile `mode`, `dynamic` shapes |
| `inference` | per-call defaults (`resize`, `max_matches`, `threshold`, sampling). Changing them needs no rebuild |
| `components` | for pipelines, one config per component (`extractor`, `matcher`) |

## Layers

Later layers win:

```
dataclass defaults < preset (and the presets it extends) < config file or dict < dotted overrides < keyword arguments
```

A config file is an overlay with the same sections:

```toml
# run.toml: RaCo+ALIKED in BF16, LightGlue INT8 (with preset int8), everything compiled
device = "cuda"
[engine]
kind = "compile"
[components.extractor]
preset = "bf16"
[inference]
resize = 1024
max_matches = 1024
```

## Overrides

| override | effect |
|---|---|
| `engine.kind=compile-full` | sets a value |
| `optimize.passes+=x` | appends to a list (a plain `=` replaces the list) |
| `components.matcher.optimize.attention=sage` | reaches into a pipeline component |
| `components.extractor.preset=bf16` | picks a component's preset (the pipeline's preset is every component's otherwise) |

## Errors and warnings

- Unknown keys, passes, quant configs and engines are errors.
- A pass the platform cannot run is skipped with a warning. `optimize.strict=true` raises instead.

## Inspecting a build

```python
flashim.resolve("rdd-lightglue", "fp8").to_dict()      # the config a build would use, without building
model.run_config.digest()                              # stable hash of what was asked for, for naming results
model.build_report                                     # what was actually applied
```

`flashim.resolve` without `platform=` does not depend on the machine. A build does: on a GPU without FP8 it replaces
`fp8` by the nearest other preset ([platforms.md](platforms.md)).
