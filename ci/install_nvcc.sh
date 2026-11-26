#!/usr/bin/env bash
set -euo pipefail

version=${1:?expected CUDA major.minor or 13.x}
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl g++ python3
keyring=$(mktemp --suffix=.deb)
trap 'rm -f "$keyring"' EXIT
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o "$keyring"
dpkg -i "$keyring"
apt-get update -qq
if [[ $version == 13.x ]]; then
    newest=$(apt-cache pkgnames cuda-nvcc-13- | sort -V | tail -1)
    version=${newest#cuda-nvcc-}
fi
apt-get install -y --no-install-recommends "cuda-nvcc-${version/./-}" "cuda-cudart-dev-${version/./-}"
