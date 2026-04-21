"""LeRobot third-party plugin: DiffusionPolicy with strided observation history.

Importing this package registers ``strided_diffusion`` as a policy type on
``PreTrainedConfig`` so it can be selected via
``lerobot-train --policy.type=strided_diffusion``.

The main motivation for this strided diffusion variant is to enable training with 
historical observation windows that span a longer time horizon than the stock 
``DiffusionConfig``'s consecutive frame window, without increasing the number of 
input steps and thus the memory requirements. This is achieved by overriding the 
``observation_delta_indices`` property to return a strided window instead of a 
consecutive one, while keeping the action prediction window contiguous.

IT IS REQUIRED that the training dataset's fps (same as in recording) be set in the 
``DATASET_FPS`` environment variable and the desired stride in seconds be set in the 
``STRIDE_SECONDS`` environment variable when using this policy type. The config will 
compute the corresponding integer frame stride and persist both values for use during 
inference with the ``StridedHistoryRunner``.
Example usage:
```bash
DATASET_FPS=10 STRIDE_SECONDS=1 lerobot-train --policy.type=strided_diffusion --config_path=path/to/config.json
```
or with a config file that specifies ``policy.type=strided_diffusion`` and the environment variables set as above.
"""

from .configuration_strided_diffusion import StridedDiffusionConfig
from .modeling_strided_diffusion import StridedDiffusionPolicy

__all__ = ["StridedDiffusionConfig", "StridedDiffusionPolicy"]
