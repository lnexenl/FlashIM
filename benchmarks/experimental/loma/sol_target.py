"""One loma-l extract + match_features under cudaProfilerStart/Stop with NVTX ranges, for ncu."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_dynamic_int8 import quantize_vit

import flashim

IMAGES = ("assets/megadepth_0015/101776896_21fc24b031_o.jpg", "assets/megadepth_0015/101795964_72e8f63ab1_o.jpg")
config = sys.argv[1]
with torch.inference_mode():
    if config == "bf16":
        p = flashim.load("loma-l", "bf16")
    else:
        p = flashim.load("loma-l", "int8", overrides=["components.matcher.optimize.attention=sage-fp8"])
        quantize_vit(p.extractor)
        for block in p.extractor.descriptor.encoder.dinov2.blocks:
            block.attn.sdpa_impl = "sage-fp8"
    ext, m = p.extractor, p.matcher
    a, b = (ext.extract(i, resize=1024, top_k=4096) for i in IMAGES)
    for _ in range(3):
        ext.extract(IMAGES[0], resize=1024, top_k=4096)
        m.match_features(a, b)
    torch.cuda.synchronize()
    parts = {"det_enc": ext.detector.encoder, "det_dec": ext.detector.decoder, "vgg": ext.descriptor.encoder.vgg,
             "vit": ext.descriptor.encoder.dinov2, "desc_dec": ext.descriptor.decoder}
    for name, mod in parts.items():
        mod.register_forward_pre_hook(lambda *_, n=name: (torch.cuda.nvtx.range_push(n), None)[1])
        mod.register_forward_hook(lambda *_: (torch.cuda.nvtx.range_pop(), None)[1])
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("extract")
    ext.extract(IMAGES[0], resize=1024, top_k=4096)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("match")
    m.match_features(a, b)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
