"""Cost and pixel fidelity of the image-path decoders of ``load_image`` on MegaDepth-1500 JPEGs.

Per image, the whole ``load_image(path, device, decoder)`` call (file read, decode, host -> device copy, uint8 -> float
/ 255, synchronized) for each decoder, best of ``--repeats`` consecutive calls with the file in the page cache, plus
torchvision's CPU ``decode_jpeg`` (libjpeg-turbo, same pixels as PIL) and the chroma post-step of "nvjpeg" alone; then
the decoded pixels against PIL's, grouped by chroma subsampling.

    python benchmarks/speed/image_decode.py --images 48 --out results/jpeg/decode_l20.json
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, JpegImagePlugin
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from flashim import ops
from flashim.image_io import chroma_layout
from flashim.models.romav2.model import IMAGE_DECODERS, load_image
from flashim.paths import DATA

SCENES = ("0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz", "0022_0.3_0.5.npz", "0022_0.5_0.7.npz")
SAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "other"}


def mega1500_images(root: Path) -> list[Path]:
    paths = set()
    for scene in SCENES:
        data = np.load(root / scene, allow_pickle=True)
        for (i, j), *_ in data["pair_infos"]:
            paths.update((data["image_paths"][i], data["image_paths"][j]))
    return [root / p for p in sorted(paths)]


def _timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=48, help="images timed (evenly spaced over the 806)")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--fidelity-images", type=int, default=0, help="images compared to PIL (0 = all 806)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    device = torch.device("cuda")
    paths = mega1500_images(DATA / "megadepth")
    timed = paths[:: max(1, len(paths) // args.images)][: args.images]
    decoders = list(IMAGE_DECODERS)

    def tv_cpu(p):
        return decode_jpeg(read_file(str(p)), mode=ImageReadMode.RGB).to(device)[None].float() / 255.0

    def post_step(p):
        layout = chroma_layout(Path(p).read_bytes())
        rgb = decode_jpeg(read_file(str(p)), mode=ImageReadMode.RGB, device=device)
        return lambda: ops.jpeg_fancy_upsample(rgb, layout) if layout else rgb

    variants = {d: (lambda p, d=d: load_image(p, device, d)) for d in decoders} | {"torchvision-cpu": tv_cpu}
    for fn in variants.values():  # warm up (nvJPEG handle, CUDA context, page cache)
        for p in timed[:4]:
            fn(p)
    times: dict[str, list[float]] = {k: [] for k in [*variants, "nvjpeg post-step"]}
    for p in timed:
        for k, fn in variants.items():
            times[k].append(min(_timed(lambda fn=fn, p=p: fn(p)) for _ in range(args.repeats)))
        step = post_step(p)
        times["nvjpeg post-step"].append(min(_timed(step) for _ in range(args.repeats)))
    summary = {k: {"median_ms": float(np.median(t)), "mean_ms": float(np.mean(t)), "max_ms": float(np.max(t))}
               for k, t in times.items()}
    megapixels = [Image.open(p).size for p in timed]
    megapixels = float(np.median([w * h / 1e6 for w, h in megapixels]))

    fidelity: dict[str, dict[str, list]] = {d: collections.defaultdict(list) for d in decoders if d != "pil"}
    for p in paths[: args.fidelity_images or None]:
        sampling = SAMPLING.get(JpegImagePlugin.get_sampling(Image.open(p)), "other")
        ref = (load_image(p, device, "pil") * 255).round()
        for d in fidelity:
            diff = ((load_image(p, device, d) * 255).round() - ref).abs()
            fidelity[d][sampling].append((float(diff.max()), float(diff.mean()), float((diff > 0).float().mean())))
            fidelity[d]["all"].append(fidelity[d][sampling][-1])
    pixel = {
        d: {
            s: {"images": len(v), "max": max(x[0] for x in v), "mean_abs": float(np.mean([x[1] for x in v])),
                "frac_differing": float(np.mean([x[2] for x in v]))}
            for s, v in sorted(groups.items())
        }
        for d, groups in fidelity.items()
    }
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "images_timed": len(timed),
              "median_megapixels": megapixels, "load_image_ms": summary, "pixels_vs_pil": pixel}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
