import torch
import torch.nn.functional as F

from flashim.utils import grid_sample


def test_grid_sample_is_fp32_for_reduced_precision_cpu_inputs():
    torch.manual_seed(0)
    x, grid = torch.randn(1, 4, 8, 8), torch.rand(1, 5, 1, 2) * 2 - 1
    kwargs = {"mode": "bilinear", "align_corners": True}
    assert torch.equal(grid_sample(x, grid, **kwargs), F.grid_sample(x, grid, **kwargs))
    out = grid_sample(x.bfloat16(), grid.bfloat16(), **kwargs)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, F.grid_sample(x.bfloat16().float(), grid.bfloat16().float(), **kwargs).bfloat16())


def test_grid_sample_survives_large_bf16_cpu_inputs():
    """torch 2.11's own BF16 CPU kernel segfaults on these within a few calls."""
    x, grid = torch.randn(1, 16, 704, 1024).bfloat16(), (torch.rand(1, 512, 1, 2) * 2 - 1).bfloat16()
    for _ in range(5):
        grid_sample(x, grid, mode="bilinear", align_corners=True)
