"""Fast sampling draws the same candidates as the reference path; with the reference KDE pinned
it is bit-identical (matches, confidences, precisions)."""

import pytest
import torch

from flashim import ops

from ...conftest import ASSETS

pytestmark = [pytest.mark.cuda, pytest.mark.weights, pytest.mark.slow]


@pytest.mark.parametrize("setting", ["turbo", "mega1500"])
def test_fast_sampling_matches_reference(setting, fp32_highest):
    from flashim.models.romav2.build import load_romav2

    model = load_romav2("fp8", setting=setting, engine="eager")
    result = model.match(ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")

    def draw(fast: bool):
        model.fast_sampling = fast
        return model.sample(result, 2000, generator=torch.Generator("cuda").manual_seed(0))

    reference = draw(False)
    ops.configure("kde_density:reference")
    try:
        fast = draw(True)
    finally:
        ops.configure(None)
    for a, b in zip(reference, fast):
        assert (a is None and b is None) or torch.equal(a, b)


def test_match_num_samples_returns_pixel_keypoints(fp32_highest):
    from PIL import Image

    from flashim.models.romav2.build import load_romav2

    model = load_romav2("fp8", setting="turbo", engine="eager")
    model.fast_sampling = True
    path_a, path_b = ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"
    result = model.match(path_a, path_b, num_samples=1000)
    (w_a, h_a), (w_b, h_b) = Image.open(path_a).size, Image.open(path_b).size
    assert result.keypoints_A.shape == result.keypoints_B.shape == (1000, 2)
    for kp, (w, h) in ((result.keypoints_A, (w_a, h_a)), (result.keypoints_B, (w_b, h_b))):
        assert (kp[:, 0] >= 0).all() and (kp[:, 0] <= w).all() and (kp[:, 1] >= 0).all() and (kp[:, 1] <= h).all()
