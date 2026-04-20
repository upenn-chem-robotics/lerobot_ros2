"""Configuration for DiffusionPolicy with uniform-stride observation history.

The only behavioural differences vs. stock ``DiffusionConfig`` are the two
override properties below: ``observation_delta_indices`` returns a strided
window ``[-stride_frames*(n-1), ..., -stride_frames, 0]`` instead of the stock
consecutive ``[1-n, ..., 0]``, and ``action_delta_indices`` is pinned to the
stock consecutive range so the diffusion head still predicts a contiguous
action chunk (this matches ``DiffusionConfig.action_delta_indices`` but we
restate it for clarity).

``stride_frames`` is derived from ``stride_seconds * fps``; both values are
persisted in ``config.json`` via draccus and consumed by the inference-time
``StridedHistoryRunner`` (see ``lerobot_tools.strided_history``). ``fps`` must
match the training dataset's fps -- ``lerobot.datasets.factory.resolve_delta_timestamps``
divides our integer frame offsets by ``ds_meta.fps`` to get seconds, which is
only correct when ``self.fps == ds_meta.fps``.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("strided_diffusion")
@dataclass
class StridedDiffusionConfig(DiffusionConfig):
    stride_seconds: float = 1.0
    fps: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        frames = self.stride_seconds * self.fps
        if abs(round(frames) - frames) > 1e-6:
            raise ValueError(
                "stride_seconds * fps must be an integer number of frames; "
                f"got stride_seconds={self.stride_seconds}, fps={self.fps} -> {frames} frames."
            )

    @property
    def stride_frames(self) -> int:
        return int(round(self.stride_seconds * self.fps))

    @property
    def observation_delta_indices(self) -> list:
        step = self.stride_frames
        return [-step * k for k in reversed(range(self.n_obs_steps))]

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
