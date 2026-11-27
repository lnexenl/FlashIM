"""Lower representative depthwise and GEMM launches to CUDA binaries without a GPU."""

from tilelang import tvm
from tilelang.engine import lower

from flashim.backends.tilelang import _conv_kernels as conv
from flashim.backends.tilelang import _gemm_kernels as gemm


def main():
    for arch in ("sm_90a", "sm_100a", "sm_110a", "sm_120a"):
        with tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
            kernels = [
                fn.get_tir(1, 32, 64, 64, conv._tile_h(32))
                for fn in (conv._build_depthwise5x5_relu_int8, conv._build_depthwise5x5_relu_mxfp8)
            ]
            if arch != "sm_90a":
                chunk = gemm.pick_chunk(2016, gemm.BLOCK_M, 64)
                kernels.append(gemm._build_mxfp8_gemm.get_tir(64, 64, 2016, 64, chunk))
            for kernel in kernels:
                lower(kernel, target=target, enable_device_compile=True)
            print(f"{arch}: compiled {len(kernels)} TileLang specializations", flush=True)


if __name__ == "__main__":
    main()
