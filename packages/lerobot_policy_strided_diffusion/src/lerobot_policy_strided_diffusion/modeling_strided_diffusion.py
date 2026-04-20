"""DiffusionPolicy variant that advertises a strided observation history.

The training and inference math is entirely inherited from
``DiffusionPolicy``; the subclass exists only so that
``lerobot.policies.factory._get_policy_cls_from_policy_name`` can resolve
``strided_diffusion`` -> ``StridedDiffusionPolicy`` by the standard
``XYZConfig`` -> ``XYZPolicy`` naming convention.
"""

from __future__ import annotations

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from .configuration_strided_diffusion import StridedDiffusionConfig


class StridedDiffusionPolicy(DiffusionPolicy):
    config_class = StridedDiffusionConfig
    name = "strided_diffusion"
