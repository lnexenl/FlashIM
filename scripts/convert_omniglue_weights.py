"""Reference TensorFlow-based converter for OmniGlue (``og_export``) and SuperPoint ``sp_v6`` SavedModels.

Run with a Python environment that has TensorFlow installed (e.g. ``~/work/oracles/omniglue/.venv/bin/python``)
to verify that FlashIM's pure-Python TensorBundle reader (``flashim/models/omniglue/tf_bundle.py``) produces
bit-identical tensors to ``tf.train.load_checkpoint`` and ``tf.saved_model.load``, or to export standalone
``.safetensors`` / ``.npz`` state dicts.

Example::

    ~/work/oracles/omniglue/.venv/bin/python scripts/convert_omniglue_weights.py \\
        --og-export ~/work/oracles/omniglue/og_export \\
        --sp-v6 ~/work/oracles/omniglue/sp_v6_extract/sp_v6 \\
        --verify-reader ~/.cache/torch/hub/checkpoints/og_export.zip ~/.cache/torch/hub/checkpoints/sp_v6.tgz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def convert_omniglue_tf(og_export_dir: str | Path) -> dict[str, np.ndarray]:
    """Load OmniGlue SavedModel variables via TensorFlow and map them to FlashIM parameter names."""
    import tensorflow as tf

    model = tf.saved_model.load(str(og_export_dir))
    fn = model.signatures["serving_default"]
    by_name = {v.name: v.numpy() for v in fn.variables}

    state: dict[str, np.ndarray] = {}
    # Keypoint encoder: 4 x (Linear + BN + ReLU) + Linear(256, 256)
    for k in range(4):
        sfx = "" if k == 0 else f"_{k}"
        state[f"kenc.layers.{k}.linear.weight"] = by_name[f"dense{sfx}/kernel:0"].T.copy()
        state[f"kenc.layers.{k}.linear.bias"] = by_name[f"dense{sfx}/bias:0"].copy()
        state[f"kenc.layers.{k}.bn.weight"] = by_name[f"batch_normalization{sfx}/gamma:0"].copy()
        state[f"kenc.layers.{k}.bn.bias"] = by_name[f"batch_normalization{sfx}/beta:0"].copy()
        state[f"kenc.layers.{k}.bn.running_mean"] = by_name[f"batch_normalization{sfx}/moving_mean:0"].copy()
        state[f"kenc.layers.{k}.bn.running_var"] = by_name[f"batch_normalization{sfx}/moving_variance:0"].copy()
    state["kenc.out.weight"] = by_name["dense_4/kernel:0"].T.copy()
    state["kenc.out.bias"] = by_name["dense_4/bias:0"].copy()

    # 18 attentional aggregation layers (alternating 9 self-attention and 9 DINO-masked cross-attention)
    for layer_idx in range(18):
        d0 = 5 + 6 * layer_idx
        bn = 4 + layer_idx
        prefix = f"gnn.layers.{layer_idx}"
        for proj_i, proj_name in enumerate(("q_proj", "k_proj", "v_proj", "out_proj")):
            state[f"{prefix}.attn.{proj_name}.weight"] = by_name[f"dense_{d0 + proj_i}/kernel:0"].T.copy()
            state[f"{prefix}.attn.{proj_name}.bias"] = by_name[f"dense_{d0 + proj_i}/bias:0"].copy()
        state[f"{prefix}.mlp.0.weight"] = by_name[f"dense_{d0 + 4}/kernel:0"].T.copy()
        state[f"{prefix}.mlp.0.bias"] = by_name[f"dense_{d0 + 4}/bias:0"].copy()
        state[f"{prefix}.mlp.1.weight"] = by_name[f"batch_normalization_{bn}/gamma:0"].copy()
        state[f"{prefix}.mlp.1.bias"] = by_name[f"batch_normalization_{bn}/beta:0"].copy()
        state[f"{prefix}.mlp.1.running_mean"] = by_name[f"batch_normalization_{bn}/moving_mean:0"].copy()
        state[f"{prefix}.mlp.1.running_var"] = by_name[f"batch_normalization_{bn}/moving_variance:0"].copy()
        state[f"{prefix}.mlp.3.weight"] = by_name[f"dense_{d0 + 5}/kernel:0"].T.copy()
        state[f"{prefix}.mlp.3.bias"] = by_name[f"dense_{d0 + 5}/bias:0"].copy()

    # Final projection + Sinkhorn dustbin score
    state["final_proj.weight"] = by_name["dense_113/kernel:0"].T.copy()
    state["final_proj.bias"] = by_name["dense_113/bias:0"].copy()
    state["bin_score"] = by_name["Variable:0"].copy()
    return state


SP_V6_CONV_MAP = (
    ("superpoint/vgg/conv1_1", "conv1a", "bn1a"),
    ("superpoint/vgg/conv1_2", "conv1b", "bn1b"),
    ("superpoint/vgg/conv2_1", "conv2a", "bn2a"),
    ("superpoint/vgg/conv2_2", "conv2b", "bn2b"),
    ("superpoint/vgg/conv3_1", "conv3a", "bn3a"),
    ("superpoint/vgg/conv3_2", "conv3b", "bn3b"),
    ("superpoint/vgg/conv4_1", "conv4a", "bn4a"),
    ("superpoint/vgg/conv4_2", "conv4b", "bn4b"),
    ("superpoint/detector/conv1", "convPa", "bnPa"),
    ("superpoint/detector/conv2", "convPb", "bnPb"),
    ("superpoint/descriptor/conv1", "convDa", "bnDa"),
    ("superpoint/descriptor/conv2", "convDb", "bnDb"),
)


def convert_sp_v6_tf(sp_v6_dir: str | Path) -> dict[str, np.ndarray]:
    """Load rpautrat/SuperPoint ``sp_v6`` TF1 SavedModel variables via TensorFlow."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(str(Path(sp_v6_dir) / "variables" / "variables"))
    state: dict[str, np.ndarray] = {}
    for tf_prefix, conv_name, bn_name in SP_V6_CONV_MAP:
        # TF Conv2D kernel [kH, kW, Cin, Cout] -> PyTorch [Cout, Cin, kH, kW]
        kernel = reader.get_tensor(f"{tf_prefix}/conv/kernel")
        state[f"{conv_name}.weight"] = np.transpose(kernel, (3, 2, 0, 1)).copy()
        state[f"{conv_name}.bias"] = reader.get_tensor(f"{tf_prefix}/conv/bias").copy()
        state[f"{bn_name}.weight"] = reader.get_tensor(f"{tf_prefix}/bn/gamma").copy()
        state[f"{bn_name}.bias"] = reader.get_tensor(f"{tf_prefix}/bn/beta").copy()
        state[f"{bn_name}.running_mean"] = reader.get_tensor(f"{tf_prefix}/bn/moving_mean").copy()
        state[f"{bn_name}.running_var"] = reader.get_tensor(f"{tf_prefix}/bn/moving_variance").copy()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--og-export", type=Path, help="Path to unzipped og_export SavedModel directory")
    parser.add_argument("--sp-v6", type=Path, help="Path to extracted sp_v6 SavedModel directory")
    parser.add_argument("--out-og", type=Path, help="Optional output .npz for OmniGlue weights")
    parser.add_argument("--out-sp", type=Path, help="Optional output .npz for sp_v6 weights")
    parser.add_argument("--dump-hashes", action="store_true", help="Print SHA-256 of every converted tensor")
    args = parser.parse_args()

    for label, src, fn, out_path in (
        ("omniglue", args.og_export, convert_omniglue_tf, args.out_og),
        ("sp_v6", args.sp_v6, convert_sp_v6_tf, args.out_sp),
    ):
        if src is None:
            continue
        state = fn(src)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **state)
            print(f"[{label}] wrote {len(state)} tensors to {out_path}")
        if args.dump_hashes:
            hashes = {
                k: [list(v.shape), str(v.dtype), hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()]
                for k, v in sorted(state.items())
            }
            print(json.dumps({label: hashes}))


if __name__ == "__main__":
    main()
