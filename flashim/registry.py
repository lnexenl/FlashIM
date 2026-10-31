"""Model registry behind :func:`flashim.load`: each model registers its :class:`~flashim.build.ModelSpec` (imported
lazily)."""

from __future__ import annotations

from flashim.build import register_spec

SPECS = {
    "romav2": "flashim.models.romav2.spec:SPEC",
    "rdd": "flashim.models.rdd.spec:SPEC",
    "lightglue": "flashim.models.lightglue.spec:SPEC",
    "aliked": "flashim.models.aliked.spec:SPEC",
    "raco": "flashim.models.raco.spec:SPEC",
    "raco-aliked": "flashim.models.raco.spec:RACO_ALIKED",
    "rdd-lightglue": "flashim.models.lightglue.spec:RDD_LIGHTGLUE",
    "aliked-lightglue": "flashim.models.lightglue.spec:ALIKED_LIGHTGLUE",
    "raco-aliked-lightglue": "flashim.models.lightglue.spec:RACO_ALIKED_LIGHTGLUE",
    "superpoint": "flashim.models.superpoint.spec:SPEC",
    "alike": "flashim.models.alike.spec:SPEC",
    "disk": "flashim.models.disk.spec:SPEC",
    "xfeat": "flashim.models.xfeat.spec:SPEC",
    "dedode": "flashim.models.dedode.spec:SPEC",
    "dedode-g": "flashim.models.dedode.spec:SPEC_G",
    "dad": "flashim.models.dad.spec:SPEC",
    "ripe": "flashim.models.ripe.spec:SPEC",
    "loftr": "flashim.models.loftr.spec:SPEC",
    "sift": "flashim.models.sift.spec:SPEC",
    "orb": "flashim.models.orb.spec:SPEC",
    "xfeat-lightglue": "flashim.models.lightglue.spec:XFEAT_LIGHTGLUE",
    "superpoint-lightglue": "flashim.models.lightglue.spec:SUPERPOINT_LIGHTGLUE",
    "disk-lightglue": "flashim.models.lightglue.spec:DISK_LIGHTGLUE",
    "loma-matcher": "flashim.models.loma.matcher_spec:SPEC",
    "loma-extractor": "flashim.models.loma.extractor_spec:SPEC",
    "loma": "flashim.models.loma.spec:LOMA",
}

# RaCo keypoints + the descriptors of any model that can describe given keypoints (raco-superpoint, raco-disk, ...).
for _descriptor in ("superpoint", "alike", "disk", "xfeat", "dedode", "dedode-g", "dad", "ripe", "rdd"):
    SPECS[f"raco-{_descriptor}"] = f"flashim.models.raco.spec:RACO_{_descriptor.upper().replace('-', '_')}"
# ... and matched by LightGlue weights of the descriptor model (not trained on RaCo's keypoints).
for _descriptor in ("superpoint", "disk", "xfeat", "rdd"):
    SPECS[f"raco-{_descriptor}-lightglue"] = (
        f"flashim.models.lightglue.spec:RACO_{_descriptor.upper()}_LIGHTGLUE"
    )
# LoMa's pipelines, one per variant (``loma`` = ``loma-b``).
for _variant in ("b128", "b", "l", "g", "r"):
    SPECS[f"loma-{_variant}"] = f"flashim.models.loma.spec:LOMA_{_variant.upper()}"

for _name, _target in SPECS.items():
    register_spec(_name, _target)

