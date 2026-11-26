# Checkpoints

Weights are not shipped with FlashIM. Every model downloads its upstream release on first use and checks it against a
pinned SHA-256.

## Lookup order

1. The `checkpoint` config key (`overrides={"checkpoint": path}`), when given.
2. The model's environment variable, `FLASHIM_<MODEL>_CHECKPOINT` (for example `FLASHIM_RDD_CHECKPOINT`), when set.
3. The torch hub checkpoint folder: `$TORCH_HOME/hub/checkpoints`, by default `~/.cache/torch/hub/checkpoints`. A file
   already there is used as is.
4. A download into that folder of the release pinned by URL and SHA-256 in `flashim/models/<model>/weights.py`.

Only downloads are hashed. A path given in step 1 or 2 must exist.

## Offline machines

Copy the checkpoint folder over, or point the `checkpoint` key or the environment variable at a local file.

## Host-specific notes

- **RIPE++ (`ripepp-aachen` and `ripepp-scared`)**: `ripe++.ckpt` (default) and `ripe++lightglue.tar` download over
  standard TLS from the authors' Hugging Face space (`JohannesK14/RIPE`). The two specialist variants
  (`ripe++_tokyo_megadepth.ckpt` and `ripe++_scared.ckpt`) are only hosted at
  `https://cvg.hhi.fraunhofer.de/RIPE%2B%2B/`, which omits the `GEANT TLS ECC 1` intermediate certificate
  (`http://crt.harica.gr/HARICA-GEANT-TLS-E1.cer`, chaining to `HARICA TLS ECC Root CA 2021`). FlashIM does not disable
  TLS verification; download those files with the intermediate supplied (e.g. `curl --cacert ...`) and place them in
  `$TORCH_HOME/hub/checkpoints/` or pass `checkpoint=` / `$FLASHIM_RIPEPP_CHECKPOINT`.

## Calibration tables

The calibration tables of the quantized presets ship in the repository, in `flashim/models/<model>/calibration/`.
