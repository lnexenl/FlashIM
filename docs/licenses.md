# Model licenses

`flashim.model_info(name)` reports the recorded licenses of each registered model's default configuration,
including every pipeline component. The code column covers model-specific code and its upstream sources.
The [package license](../README.md#license) covers distribution of FlashIM itself. Weight licenses apply to the
downloaded checkpoints. Selecting different weights can change the terms. For example, standalone SuperPoint
uses Magic Leap's `v1` weights by default, while OmniGlue uses the MIT-licensed `sp_v6` release.

```python
>>> import flashim
>>> info = flashim.model_info("ufm")
>>> info.weights_licenses, info.commercial_ok
(('CC-BY-NC-SA-4.0',), False)
```

`commercial_ok` is true when all recorded code and weight licenses are Apache-2.0, BSD-3-Clause or MIT.
Weights that the publisher distributes in its licensed repository or releases inherit that license unless
upstream specifies different terms. Each model's `weights.py` records the basis. `NOASSERTION` marks a
checkpoint or component whose terms could not be established, including externally hosted weights without
a grant covering them.
An unknown license, a non-commercial license or a custom agreement requiring review yields false.

Magic Leap's SuperPoint and SuperGlue licenses restrict use to non-commercial research. The Fraunhofer
academic-use license v2.0 and CC-BY-NC-SA-4.0 also restrict commercial use. FlashIM logs a warning before the
first download of these weights per model family in a process; it reaches stderr unless the application configures
Python logging itself. See the [third-party notices](../THIRD_PARTY_NOTICES.md)
and the [DINOv3 License Agreement](../licenses/DINOv3-License.md) for the upstream terms.

LoMa's default checkpoints inherit MIT from the publisher's weight release. Its README also identifies separate
`commercial_loma_*` checkpoints trained exclusively on commercially licensed data. The table describes the
checkpoint license. FlashIM's LoMa defaults use the original training set. See [LoMa's README](https://github.com/davnords/LoMa#pretrained-weights).

## Registered models

| Model | Model / upstream code | Weights | `commercial_ok` |
| --- | --- | --- | --- |
| `alike` | BSD-3-Clause | BSD-3-Clause | true |
| `aliked` | Apache-2.0, BSD-3-Clause | BSD-3-Clause | true |
| `aliked-lightglue` | Apache-2.0, BSD-3-Clause | Apache-2.0, BSD-3-Clause | true |
| `dad` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `dedode` | MIT | MIT | true |
| `dedode-g` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `disk` | Apache-2.0 | Apache-2.0 | true |
| `disk-lightglue` | Apache-2.0 | Apache-2.0 | true |
| `dust3r` | CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 | false |
| `eloftr` | Apache-2.0 | Apache-2.0 | true |
| `flexglue` | MIT | LicenseRef-DINOv3, MIT | false |
| `lightglue` | Apache-2.0 | NOASSERTION | false |
| `loftr` | Apache-2.0 | Apache-2.0 | true |
| `loma` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `loma-b` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `loma-b128` | Apache-2.0, MIT | MIT | true |
| `loma-extractor` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `loma-g` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `loma-l` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `loma-matcher` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `loma-r` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `mast3r` | CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 | false |
| `minima-loftr` | Apache-2.0 | Apache-2.0 | true |
| `minima-roma` | Apache-2.0, MIT | Apache-2.0 | true |
| `omniglue` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `omniglue-extractor` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `omniglue-matcher` | Apache-2.0 | Apache-2.0 | true |
| `orb` | Apache-2.0 | No weights | true |
| `raco` | Apache-2.0 | Apache-2.0 | true |
| `raco-alike` | Apache-2.0, BSD-3-Clause | Apache-2.0, BSD-3-Clause | true |
| `raco-aliked` | Apache-2.0, BSD-3-Clause | Apache-2.0, BSD-3-Clause | true |
| `raco-aliked-lightglue` | Apache-2.0, BSD-3-Clause | Apache-2.0, BSD-3-Clause | true |
| `raco-dad` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `raco-dedode` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `raco-dedode-g` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `raco-disk` | Apache-2.0 | Apache-2.0 | true |
| `raco-disk-lightglue` | Apache-2.0 | Apache-2.0 | true |
| `raco-rdd` | Apache-2.0 | Apache-2.0, NOASSERTION | false |
| `raco-rdd-lightglue` | Apache-2.0 | Apache-2.0, NOASSERTION | false |
| `raco-ripe` | Apache-2.0, LicenseRef-Fraunhofer-Academic-2.0 | Apache-2.0, LicenseRef-Fraunhofer-Academic-2.0 | false |
| `raco-superpoint` | Apache-2.0 | Apache-2.0, LicenseRef-MagicLeap-SuperPoint | false |
| `raco-superpoint-lightglue` | Apache-2.0 | Apache-2.0, LicenseRef-MagicLeap-SuperPoint | false |
| `raco-xfeat` | Apache-2.0 | Apache-2.0 | true |
| `raco-xfeat-lightglue` | Apache-2.0 | Apache-2.0 | true |
| `rdd` | Apache-2.0 | NOASSERTION | false |
| `rdd-lightglue` | Apache-2.0 | NOASSERTION | false |
| `ripe` | LicenseRef-Fraunhofer-Academic-2.0 | LicenseRef-Fraunhofer-Academic-2.0 | false |
| `ripepp` | Apache-2.0, LicenseRef-Fraunhofer-Academic-2.0, MIT | LicenseRef-Fraunhofer-Academic-2.0 | false |
| `ripepp-aachen` | Apache-2.0, LicenseRef-Fraunhofer-Academic-2.0, MIT | LicenseRef-Fraunhofer-Academic-2.0 | false |
| `ripepp-lightglue` | Apache-2.0, LicenseRef-Fraunhofer-Academic-2.0, MIT | LicenseRef-Fraunhofer-Academic-2.0 | false |
| `ripepp-scared` | Apache-2.0, LicenseRef-Fraunhofer-Academic-2.0, MIT | LicenseRef-Fraunhofer-Academic-2.0 | false |
| `roma` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `roma-indoor` | Apache-2.0, MIT | Apache-2.0, MIT | true |
| `romav2` | LicenseRef-DINOv3, MIT | LicenseRef-DINOv3, NOASSERTION | false |
| `sift` | Apache-2.0 | No weights | true |
| `superglue` | Apache-2.0 | LicenseRef-MagicLeap-SuperGlue | false |
| `superpoint` | Apache-2.0 | LicenseRef-MagicLeap-SuperPoint | false |
| `superpoint-lightglue` | Apache-2.0 | Apache-2.0, LicenseRef-MagicLeap-SuperPoint | false |
| `superpoint-superglue` | Apache-2.0 | LicenseRef-MagicLeap-SuperGlue, LicenseRef-MagicLeap-SuperPoint | false |
| `tiny-roma` | Apache-2.0, MIT | MIT | true |
| `traqpoint` | Apache-2.0 | Apache-2.0, LicenseRef-DINOv3 | false |
| `ufm` | Apache-2.0, BSD-3-Clause, CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 | false |
| `xfeat` | Apache-2.0 | Apache-2.0 | true |
| `xfeat-lightglue` | Apache-2.0 | Apache-2.0 | true |
| `xoftr` | Apache-2.0 | Apache-2.0 | true |
