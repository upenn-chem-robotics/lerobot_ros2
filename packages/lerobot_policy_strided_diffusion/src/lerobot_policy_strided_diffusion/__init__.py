"""LeRobot third-party plugin: DiffusionPolicy with strided observation history.

Importing this package registers ``strided_diffusion`` as a policy type on
``PreTrainedConfig`` so it can be selected via
``lerobot-train --policy.type=strided_diffusion``.
"""

from .configuration_strided_diffusion import StridedDiffusionConfig
from .modeling_strided_diffusion import StridedDiffusionPolicy

__all__ = ["StridedDiffusionConfig", "StridedDiffusionPolicy"]
