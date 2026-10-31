# LoMa reference implementation

`loma/` is https://github.com/davnords/LoMa @ 415caed (`src/loma/`), restricted to what `LoMa.match` imports:
`loma.py`, `types.py`, `device.py`, `io.py`, `geometry.py` (for `get_normalized_grid` / `kde`; imports `cv2` and
`einops`), `descriptor/` (DeDoDe descriptor; `transformer/` is DINOv2) and `detector/` (DaD). Not vendored:
`benchmarks/`, `cfg.py` (tyro), `features.py`, `loss.py`, `normalizers.py`, `random.py`, the demo / eval scripts.

Licenses: LoMa is MIT (`LICENSE`) except the matcher in `loma.py`, which derives from cvg/LightGlue and keeps its
Apache-2.0 license (`LICENSE_LightGlue`, upstream README "License"). `descriptor/transformer/` is Meta's DINOv2,
Apache-2.0 (`LICENSE_DINOv2`); those files keep their Meta headers.

Every vendored file is unmodified apart from its header. The patches needed to run it offline live in
`loma_oracle.py` (FlashIM code, not upstream), applied at build time instead of in the files:

- `weights_url=None` in the variant config, so `LoMa.__init__` downloads nothing; the variant checkpoint is then
  loaded from the given path with upstream's own checks (no missing keys; unexpected keys only for matcher layers
  `>= n_layers`).
- `torch.hub.load_state_dict_from_url` is replaced during construction: `DaD.__init__`'s `dad.pth` and
  `FrozenDINOv2.__init__`'s `dinov2_vitl14_pretrain.pth` are answered from the checkpoint's `_detector.*` (bit-equal
  to `dad.pth`) and `_descriptor.encoder.frozen_dinov2.dinov2_vitl14.*` (BF16-rounded `dinov2_vitl14_pretrain.pth`,
  overwritten by the checkpoint load anyway); any other URL raises.
- `mp=` sets `LoMa.Cfg.mp` (the matcher's autocast). `extractor_amp=False` additionally clears the `amp` flag of
  every VGG / ConvRefiner / FrozenDINOv2 (upstream hard-codes them on) and converts DINOv2 to FP32 before the
  checkpoint is loaded: the FP32 reference.

The model lives on `loma.device.device` (CUDA when available; `amp_dtype` BF16 on sm_80+, FP16 below).
