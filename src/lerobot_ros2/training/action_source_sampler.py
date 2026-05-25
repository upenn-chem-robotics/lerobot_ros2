"""DAgger-aware samplers that filter and reweight anchor frames by ``action_source``.

When training on a DAgger-aggregated dataset (base demonstrations + dagger
correction episodes), each dagger episode is a sequence of:

    [policy prefix (action_source=0) | human takeover (action_source=1)]

The prefix frames are kept in the dataset so the takeover-boundary anchor
gets a real "approaching-failure" observation history (via DiffusionPolicy's
``observation_delta_indices``), but they should *never* be picked as the
"current" frame ``t`` whose action chunk would otherwise contain the
policy's failure command.

Two samplers are provided:

* :class:`ActionSourceAwareEpisodeSampler` -- subclass of
  :class:`lerobot.datasets.sampler.EpisodeAwareSampler` that drops any
  anchor with ``action_source == 0`` and otherwise iterates uniformly.
* :class:`WeightedDaggerEpisodeSampler` -- subclass of the above that
  additionally re-weights anchors per epoch so that base-episode anchors
  and dagger-episode anchors hit a target sampling fraction (e.g. ensure
  dagger frames make up 25 % of training samples even though they are
  naturally 55 % of the surviving anchors).

The dataset rows themselves are *not* modified; only the sampler's anchor
list (and its sampling distribution) changes, so observations from prefix
frames remain reachable as history context for nearby human-takeover
anchors.
"""

from __future__ import annotations

from typing import Iterable, Iterator

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


class WeightedDaggerEpisodeSampler(ActionSourceAwareEpisodeSampler):
    """Action-source-aware sampler with weighted base/dagger sampling per epoch.

    After the parent has filtered anchors to ``action_source == 1``, this
    sampler classifies each surviving anchor as belonging to a *base*
    episode (every frame in the episode has ``action_source == 1``) or a
    *dagger* episode (the episode contains at least one ``action_source == 0``
    frame). It then assigns a per-anchor weight so that, in expectation,
    ``dagger_fraction`` of the ``num_samples`` indices yielded per epoch
    come from dagger episodes.

    Per epoch, ``__iter__`` draws ``num_samples`` anchors with replacement
    using :func:`torch.multinomial`. ``num_samples`` defaults to the number
    of surviving anchors so training-time per epoch is unchanged.

    If ``dagger_fraction`` is ``None``, the sampler degrades to its parent's
    uniform iteration (same indices, same order semantics).

    Args:
        dataset_from_indices: per-episode start row index (forwarded).
        dataset_to_indices: per-episode end row index (forwarded).
        action_source: per-frame ``action_source`` array (forwarded to parent).
        is_dagger_per_frame: 1-D ``bool`` array of length ``num_frames``.
            ``True`` iff the row's episode contains any ``action_source == 0``
            frame. Required when ``dagger_fraction`` is set.
        dagger_fraction: desired fraction of dagger-episode samples per
            epoch in ``[0, 1]``. If ``None``, fall back to uniform sampling.
        num_samples: number of anchors yielded per epoch. Defaults to
            ``len(self.indices)`` (i.e. preserves the parent's epoch size).
        generator: optional :class:`torch.Generator` for reproducible
            weighted draws. Defaults to PyTorch's global RNG.
        episode_indices_to_use, drop_n_first_frames, drop_n_last_frames,
        shuffle: forwarded to parent.
    """

    def __init__(
        self,
        dataset_from_indices: Iterable[int],
        dataset_to_indices: Iterable[int],
        action_source: np.ndarray | torch.Tensor | list[int],
        is_dagger_per_frame: np.ndarray | torch.Tensor | list[bool] | None = None,
        dagger_fraction: float | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ) -> None:
        super().__init__(
            dataset_from_indices,
            dataset_to_indices,
            action_source=action_source,
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=shuffle,
        )

        self.dagger_fraction = dagger_fraction
        self.num_samples = int(num_samples) if num_samples is not None else len(self.indices)
        self.generator = generator
        self.n_base_anchors = len(self.indices)
        self.n_dagger_anchors = 0
        self.weights: torch.Tensor | None = None

        if dagger_fraction is None:
            return

        if not (0.0 <= float(dagger_fraction) <= 1.0):
            raise ValueError(
                f"dagger_fraction must be in [0, 1], got {dagger_fraction!r}"
            )
        if is_dagger_per_frame is None:
            raise ValueError(
                "is_dagger_per_frame is required when dagger_fraction is set"
            )

        if isinstance(is_dagger_per_frame, torch.Tensor):
            is_dagger_per_frame = is_dagger_per_frame.cpu().numpy()
        is_dagger_arr = np.asarray(is_dagger_per_frame).reshape(-1).astype(bool)

        anchor_idx = np.asarray(self.indices, dtype=np.int64)
        if anchor_idx.size and anchor_idx.max() >= is_dagger_arr.size:
            raise ValueError(
                f"is_dagger_per_frame has length {is_dagger_arr.size} but anchor "
                f"indices reach {int(anchor_idx.max())}; arrays must be aligned"
            )
        anchor_is_dagger = is_dagger_arr[anchor_idx]

        n_base = int((~anchor_is_dagger).sum())
        n_dagger = int(anchor_is_dagger.sum())
        self.n_base_anchors = n_base
        self.n_dagger_anchors = n_dagger

        if dagger_fraction > 0 and n_dagger == 0:
            raise ValueError(
                "dagger_fraction > 0 was requested but no surviving anchors come "
                "from dagger episodes. Check is_dagger_per_frame."
            )
        if dagger_fraction < 1 and n_base == 0:
            raise ValueError(
                "dagger_fraction < 1 was requested but no surviving anchors come "
                "from base episodes. Check is_dagger_per_frame."
            )

        weights = torch.empty(len(self.indices), dtype=torch.double)
        if n_base > 0:
            weights[torch.from_numpy(~anchor_is_dagger)] = (
                (1.0 - float(dagger_fraction)) / n_base
            )
        if n_dagger > 0:
            weights[torch.from_numpy(anchor_is_dagger)] = (
                float(dagger_fraction) / n_dagger
            )
        self.weights = weights

    def __iter__(self) -> Iterator[int]:
        if self.weights is None:
            yield from super().__iter__()
            return
        chosen = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=True,
            generator=self.generator,
        )
        for i in chosen.tolist():
            yield self.indices[int(i)]

    def __len__(self) -> int:
        return self.num_samples if self.weights is not None else len(self.indices)
