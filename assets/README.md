# Test and benchmark images

Small images used by the parity tests and speed benchmarks; the datasets (`data/megadepth`, `data/scannet`,
`data/quantization`, `data/hardmatch`) are not in git (see `flashim/paths.py`, `FLASHIM_DATA`).

- `toronto_A.jpg`, `toronto_B.jpg`: from the [RoMaV2](https://github.com/Parskatt/RoMaV2) repository (`assets/`).
- `megadepth_0015/`: six images of MegaDepth scene 0015 (`Undistorted_SfM/0015/images`), as used by RoMaV2's data.

The files were taken from RoMaV2, which is MIT-licensed: `LICENSE-RoMaV2` is its license text (Copyright (c) 2025
Johan Edstedt). - `sacre_coeur1.jpg`, `sacre_coeur2.jpg`: from [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) `assets/` (Apache-2.0,
  `tests/oracles/raco/LICENSE`), used by the RaCo parity test.

MegaDepth is distributed under its own terms (https://www.cs.cornell.edu/projects/megadepth/).
