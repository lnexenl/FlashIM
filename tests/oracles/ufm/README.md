# UFM reference implementation

`ufm/` vendors the minimal files needed to run upstream `UniFlowMatchConfidence` (`infinity1096/UFM-Base`):

- `uniflowmatch/`: from https://github.com/UniFlowMatch/UFM (`BSD-3-Clause`, `LICENSE`).
- `uniception/`: from https://github.com/castacks/UniCeption @ `ee7fa0b5` (`BSD-3-Clause`, `LICENSE_UniCeption`;
  `libs/croco/dpt_block.py` and `pos_embed.py` carry Naver's CC BY-NC-SA 4.0 header).
- `ufm_oracle.py`: FlashIM's offline builder (stubs `jaxtyping` if absent, builds DINOv2 ViT-L/14 from the
  already-vendored `tests/oracles/dedode/DeDoDe/transformer/dinov2.py`, and loads weights from FlashIM's
  `ufm_base.safetensors` with `strict=True`).
