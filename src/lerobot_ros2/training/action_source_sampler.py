"""DAgger-aware sampler that filters anchor frames by ``action_source``.

When training on a DAgger-aggregated dataset (base demonstrations + dagger
correction episodes), each dagger episode is a sequence of:

    [policy prefix (action_source=0) | human takeover (action_source=1)]

The prefix frames are kept in the dataset so the takeover-boundary anchor
gets a real "approaching-failure" observation history (via DiffusionPolicy's
``observation_delta_indices``), but they should *never* be picked as the
"current" frame ``t`` whose action chunk would otherwise contain the
policy's failure command.

This sampler subclasses :class:`lerobot.datasets.sampler.EpisodeAwareSampler`
and, after the parent has computed valid anchors via
``drop_n_first_frames`` / ``drop_n_last_frames``, drops any anchor index
``i`` where ``action_source[i] == 0``.

The dataset rows themselves are *not* modified; only the sampler's anchor
list is filtered, so observations from prefix frames remain reachable as
history context for nearby human-takeover anchors.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler


class ActionSourceAwareEpisodeSampler(EpisodeAwareSampler):
    """``EpisodeAwareSampler`` that only emits anchors with ``action_source == 1``.

    Constructor matches :class:`EpisodeAwareSampler` exactly (so it can be
    monkey-patched in for stock ``lerobot-train``) plus one extra
    ``action_source`` keyword.

    Args:
        dataset_from_indices: per-episode start row index (forwarded to parent).
        dataset_to_indices: per-episode end row index (forwarded to parent).
        action_source: 1-D ``int64`` array of length ``dataset.num_frames``
            with ``0`` for policy-driven frames and ``1`` for human/expert
            frames. Required.
        episode_indices_to_use: optional subset of episode indices.
        drop_n_first_frames: forwarded to parent.
        drop_n_last_frames: forwarded to parent.
        shuffle: forwarded to parent.
    """

    def __init__(
        self,
        dataset_from_indices: Iterable[int],
        dataset_to_indices: Iterable[int],
        action_source: np.ndarray | torch.Tensor | list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ) -> None:
        super().__init__(
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=shuffle,
        )

        arr = np.asarray(action_source).reshape(-1).astype(np.int64)

        before = len(self.indices)
        self.indices = [int(i) for i in self.indices if int(arr[int(i)]) == 1]
        after = len(self.indices)

        if not self.indices:
            raise ValueError(
                "ActionSourceAwareEpisodeSampler: no anchors remain after filtering "
                "by action_source==1. Check that the dataset has any human-tagged "
                "frames and that drop_n_first_frames / drop_n_last_frames aren't "
                "removing them all."
            )

        self.dropped_by_action_source = before - after
