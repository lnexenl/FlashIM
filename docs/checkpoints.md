# Checkpoints

Weights are not shipped with FlashIM. Every model downloads its upstream release on first use and checks it against a
pinned SHA-256.

## Lookup order

1. `checkpoint=`, when given.
2. The model's environment variable, `FLASHIM_<MODEL>_CHECKPOINT` (for example `FLASHIM_RDD_CHECKPOINT`), when set.
3. The torch hub checkpoint folder: `$TORCH_HOME/hub/checkpoints`, by default `~/.cache/torch/hub/checkpoints`. A file
   already there is used as is.
4. A download into that folder of the release pinned by URL and SHA-256 in `flashim/models/<model>/weights.py`.

Only downloads are hashed. A path given in step 1 or 2 must exist.

## Offline machines

Copy the checkpoint folder over, or point `checkpoint=` or the environment variable at a local file.

## Calibration tables

The calibration tables of the quantized presets ship in the repository, in `flashim/models/<model>/calibration/`.
