# OmniGlue reference implementation

`omniglue/` and `third_party/dinov2/` are vendored from https://github.com/google-research/omniglue
(Apache-2.0, `LICENSE` in this directory), restricted to the inference modules imported by the parity tests:

- `omniglue/superpoint_extract.py`, `omniglue/utils.py`, `omniglue/dino_extract.py`, `omniglue/omniglue_extract.py`
  (Google LLC, Apache-2.0; `omniglue/__init__.py` trimmed so TF and PyTorch modules can be imported independently)
- `third_party/dinov2/dino.py`, `third_party/dinov2/dino_utils.py` (Meta Platforms, Apache-2.0)

Upstream ships no Python model code for the OmniGlue matcher or SuperPoint `sp_v6` — both are TensorFlow
SavedModels (`og_export.zip` and `sp_v6.tgz`). Because `tensorflow` is not installed in FlashIM's main `.venv`,
`tf_oracle.py` (FlashIM test helper, not upstream) runs the TF SavedModels in a separate TensorFlow-CPU virtual
environment and exchanges inputs/outputs via `.npz` files in a temporary directory.

To enable the TF oracle tests in `tests/parity/omniglue/test_parity.py`, set `FLASHIM_OMNIGLUE_TF_PYTHON` to a
Python interpreter with `tensorflow-cpu`, `numpy`, `opencv-python-headless`, and `Pillow` installed (default
fallback: `~/work/oracles/omniglue/.venv/bin/python`). When absent, the TF-dependent parity tests skip cleanly.
