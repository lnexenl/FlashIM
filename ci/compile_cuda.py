"""Build one fat shared library per source, without importing torch or loading a CUDA driver."""

import argparse
import runpy
import subprocess
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "flashim/backends/cuda"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument(
        "--all-supported", action="store_true", help="all supported SMs between sm_80 and sm_120"
    )
    parser.add_argument("architectures", type=int, nargs="*")
    args = parser.parse_args()
    flags = runpy.run_path(str(BACKEND / "_flags.py"))["FLAGS"]
    if bool(args.architectures) == args.all_supported:
        parser.error("supply architectures or --all-supported")
    architectures = args.architectures
    if args.all_supported:
        codes = subprocess.check_output([args.nvcc, "--list-gpu-code"], text=True).split()
        numerics = (code.removeprefix("sm_") for code in codes)
        architectures = sorted(int(code) for code in numerics if code.isdigit() and 80 <= int(code) <= 120)
        if not {80, 110, 120}.issubset(architectures):
            raise RuntimeError(f"expected an Ampere-through-Thor/Blackwell compiler, got {codes}")
    targets = [f"-gencode=arch=compute_{cc},code=sm_{cc}" for cc in architectures]
    with tempfile.TemporaryDirectory() as directory:
        for source in sorted((BACKEND / "csrc").glob("*.cu")):
            print(f"{source.name}: {architectures}", flush=True)
            output = Path(directory, source.stem + ".so")
            subprocess.run([args.nvcc, *flags, *targets, source, "-o", output], check=True)


if __name__ == "__main__":
    main()
