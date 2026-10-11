# Configuring a run

Every model is built by one generic builder from a `RunConfig` (`flashim/config.py`). A preset is a named partial
config; files, dicts and dotted overrides layer on top.

## Two ways to load

```python
# keyword API: the model's own load_* function
model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile-full")

# config API: preset + config file or dict + dotted overrides
model = flashim.load("romav2", "fp8", overrides=["engine.kind=compile-full", "arch.setting=mega1500"])
model = flashim.load("raco-aliked-lightglue", "int8", config="run.toml")   # .toml / .json (.yaml with PyYAML)
```

The keyword API is a thin layer over the same builder. It exists for `romav2`, `rdd`, `aliked`, `raco-aliked`,
`lightglue` and the RDD, ALIKED and RaCo-ALIKED LightGlue pipelines; every model has the config API. One call uses
one or the other, not both.

## Sections

| section | what it holds |
|---|---|
| top level | `model`, `variant`, `checkpoint`, `device` |
| `arch` | model-specific architecture fields. They override the checkpoint's config and change the graph |
| `precision` | compute dtype, quantization config, calibration table, per-module precision overrides |
| `optimize` | named passes (model-specific fusions), attention implementation, op overrides, `strict`, pass options |
| `engine` | `kind` (`eager`, `compile`, ...), compile `mode`, `dynamic` shapes |
| `inference` | per-call defaults (resolution, `top_k`, thresholds, sampling). Changing them needs no rebuild |
| `components` | for pipelines, one config per component (`extractor`, `matcher`) |

## Layers

Later layers win:

```
dataclass defaults < preset (and the presets it extends) < config file or dict < dotted overrides < keyword arguments
```

A config file is an overlay with the same sections:

```toml
# run.toml: RaCo+ALIKED in BF16, LightGlue INT8, everything compiled
device = "cuda"
[engine]
kind = "compile"
[components.extractor]
preset = "bf16"
[inference]
resize = 1024
top_k = 1024
```

## Overrides

| override | effect |
|---|---|
| `engine.kind=compile-full` | sets a value |
| `optimize.passes+=x` | appends to a list (a plain `=` replaces the list) |
| `components.matcher.optimize.attention=sage` | reaches into a pipeline component |
| `components.extractor.preset=bf16` | picks a component's preset. A pipeline's own preset only sets LightGlue |

## Errors and warnings

- Unknown keys, passes, quant configs and engines are errors.
- A pass the platform cannot run is skipped with a warning. `optimize.strict=true` raises instead.

## Inspecting a build

```python
flashim.resolve("rdd-lightglue", "fp8").to_dict()      # the config a build would use, without building
model.run_config.digest()                              # stable hash of what was asked for, for naming results
model.build_report                                     # what was actually applied
```

`flashim.resolve` without `platform=` does not depend on the machine. A build does: on a GPU without FP8 it replaces an
FP8 preset by its fallback ([platforms.md](platforms.md)).
