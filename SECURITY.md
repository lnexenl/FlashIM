# Security policy

## Supported versions

Only the latest published FlashIM release is supported with security fixes. Older releases may not receive fixes; update
to the latest release before reporting or deploying a fix.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's private vulnerability reporting: open the repository's
**Security** tab and choose **Report a vulnerability**. Do not disclose a suspected vulnerability in a public issue.

Private vulnerability reporting must be enabled by the repository maintainer in the repository settings. If the option is
not available, contact the maintainer through [GitHub](https://github.com/lnexenl) and request that it be enabled before
sharing sensitive details.

Include the affected release or commit, a clear description, reproduction steps or a minimal proof of concept, and the
potential impact. Please also describe any required hardware, driver, or checkpoint conditions.

## Relevant attack surface

FlashIM downloads model checkpoints from upstream publishers and verifies their pinned SHA-256 hashes before accepting a
download. Model loading uses PyTorch's `weights_only=True` mode where applicable; see
[`flashim/models/weights_util.py`](flashim/models/weights_util.py) and [`tests/unit/test_weights.py`](tests/unit/test_weights.py)
for the current guarantees and tests. Some GPU paths JIT-compile CUDA sources on the local machine, so reports involving
checkpoint files, compiler inputs, generated binaries, or the CUDA toolchain should include the relevant versions and paths.
