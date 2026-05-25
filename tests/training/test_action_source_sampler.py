"""Unit tests for the DAgger-aware samplers.

Covers both :class:`ActionSourceAwareEpisodeSampler` and the newer weighted
variant :class:`WeightedDaggerEpisodeSampler`. The tests use small synthetic
datasets so they run in <1 s and don't touch real data on disk.

Run::

    /opt/conda/envs/lerobot/bin/python -m pytest tests/training/ -q

or, since the repo doesn't ship a pytest config, simply::

    /opt/conda/envs/lerobot/bin/python tests/training/test_action_source_sampler.py
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from lerobot_ros2.training.action_source_sampler import (
    ActionSourceAwareEpisodeSampler,
    WeightedDaggerEpisodeSampler,
)


def _make_toy_dataset(
    n_base_eps: int = 3,
    n_dagger_eps: int = 2,
    base_len: int = 10,
    dagger_len: int = 12,
    policy_prefix_in_dagger: int = 3,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Build the inputs an EpisodeAwareSampler needs from a synthetic dataset.

    Returns ``(from_indices, to_indices, action_source, is_dagger_per_frame)``.

    Base episodes are pure-human (every frame ``action_source == 1``).
    Dagger episodes start with ``policy_prefix_in_dagger`` policy frames
    (``action_source == 0``) followed by ``dagger_len - prefix`` human frames.
    """
    from_indices: list[int] = []
    to_indices: list[int] = []
    action_source_list: list[int] = []
    is_dagger_list: list[bool] = []

    cursor = 0
    for _ in range(n_base_eps):
        from_indices.append(cursor)
        cursor += base_len
        to_indices.append(cursor)
        action_source_list.extend([1] * base_len)
        is_dagger_list.extend([False] * base_len)

    for _ in range(n_dagger_eps):
        from_indices.append(cursor)
        cursor += dagger_len
        to_indices.append(cursor)
        action_source_list.extend(
            [0] * policy_prefix_in_dagger + [1] * (dagger_len - policy_prefix_in_dagger)
        )
        is_dagger_list.extend([True] * dagger_len)

    action_source = np.asarray(action_source_list, dtype=np.int64)
    is_dagger = np.asarray(is_dagger_list, dtype=bool)
    return from_indices, to_indices, action_source, is_dagger


class ActionSourceAwareSamplerTests(unittest.TestCase):
    def test_filters_policy_anchors(self):
        from_idx, to_idx, action_source, _ = _make_toy_dataset()
        sampler = ActionSourceAwareEpisodeSampler(
            from_idx, to_idx, action_source=action_source
        )
        for anchor in sampler:
            self.assertEqual(int(action_source[anchor]), 1)

    def test_counts_dropped_frames(self):
        from_idx, to_idx, action_source, _ = _make_toy_dataset(
            n_base_eps=2, n_dagger_eps=3, base_len=5, dagger_len=8, policy_prefix_in_dagger=2
        )
        sampler = ActionSourceAwareEpisodeSampler(
            from_idx, to_idx, action_source=action_source
        )
        expected_kept = 2 * 5 + 3 * (8 - 2)
        self.assertEqual(len(sampler), expected_kept)
        self.assertEqual(sampler.dropped_by_action_source, 3 * 2)

    def test_drop_n_last_frames_still_applies(self):
        from_idx, to_idx, action_source, _ = _make_toy_dataset(
            n_base_eps=1, n_dagger_eps=0, base_len=10
        )
        sampler = ActionSourceAwareEpisodeSampler(
            from_idx, to_idx, action_source=action_source, drop_n_last_frames=3
        )
        self.assertEqual(len(sampler), 7)

    def test_all_policy_raises(self):
        from_idx = [0]
        to_idx = [4]
        action_source = np.array([0, 0, 0, 0], dtype=np.int64)
        with self.assertRaises(ValueError):
            ActionSourceAwareEpisodeSampler(from_idx, to_idx, action_source=action_source)


class WeightedDaggerSamplerTests(unittest.TestCase):
    def test_no_fraction_behaves_like_parent(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset()
        parent = ActionSourceAwareEpisodeSampler(
            from_idx, to_idx, action_source=action_source
        )
        weighted = WeightedDaggerEpisodeSampler(
            from_idx,
            to_idx,
            action_source=action_source,
            is_dagger_per_frame=is_dagger,
            dagger_fraction=None,
        )
        self.assertEqual(list(parent), list(weighted))
        self.assertEqual(len(parent), len(weighted))

    def test_weights_sum_to_one(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset()
        sampler = WeightedDaggerEpisodeSampler(
            from_idx,
            to_idx,
            action_source=action_source,
            is_dagger_per_frame=is_dagger,
            dagger_fraction=0.25,
        )
        self.assertAlmostEqual(float(sampler.weights.sum().item()), 1.0, places=10)

    def test_weighted_draws_hit_target_fraction(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset(
            n_base_eps=3, n_dagger_eps=2, base_len=10, dagger_len=12, policy_prefix_in_dagger=3
        )
        target = 0.25
        gen = torch.Generator()
        gen.manual_seed(0)
        sampler = WeightedDaggerEpisodeSampler(
            from_idx,
            to_idx,
            action_source=action_source,
            is_dagger_per_frame=is_dagger,
            dagger_fraction=target,
            num_samples=20_000,
            generator=gen,
        )
        drawn = list(sampler)
        # Every drawn anchor must still be action_source == 1.
        for anchor in drawn:
            self.assertEqual(int(action_source[anchor]), 1)
        # Empirical dagger fraction over a large sample should be near target.
        n_dagger = sum(1 for a in drawn if bool(is_dagger[a]))
        empirical = n_dagger / len(drawn)
        self.assertAlmostEqual(empirical, target, delta=0.01)

    def test_dagger_zero_means_only_base(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset()
        gen = torch.Generator()
        gen.manual_seed(0)
        sampler = WeightedDaggerEpisodeSampler(
            from_idx,
            to_idx,
            action_source=action_source,
            is_dagger_per_frame=is_dagger,
            dagger_fraction=0.0,
            num_samples=5_000,
            generator=gen,
        )
        for anchor in sampler:
            self.assertFalse(bool(is_dagger[anchor]))

    def test_dagger_one_means_only_dagger(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset()
        gen = torch.Generator()
        gen.manual_seed(0)
        sampler = WeightedDaggerEpisodeSampler(
            from_idx,
            to_idx,
            action_source=action_source,
            is_dagger_per_frame=is_dagger,
            dagger_fraction=1.0,
            num_samples=5_000,
            generator=gen,
        )
        for anchor in sampler:
            self.assertTrue(bool(is_dagger[anchor]))

    def test_num_samples_controls_epoch_size(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset()
        gen = torch.Generator()
        gen.manual_seed(0)
        sampler = WeightedDaggerEpisodeSampler(
            from_idx,
            to_idx,
            action_source=action_source,
            is_dagger_per_frame=is_dagger,
            dagger_fraction=0.25,
            num_samples=1234,
            generator=gen,
        )
        self.assertEqual(len(sampler), 1234)
        self.assertEqual(len(list(sampler)), 1234)

    def test_invalid_fraction_raises(self):
        from_idx, to_idx, action_source, is_dagger = _make_toy_dataset()
        with self.assertRaises(ValueError):
            WeightedDaggerEpisodeSampler(
                from_idx,
                to_idx,
                action_source=action_source,
                is_dagger_per_frame=is_dagger,
                dagger_fraction=1.5,
            )

    def test_missing_is_dagger_raises_when_fraction_set(self):
        from_idx, to_idx, action_source, _ = _make_toy_dataset()
        with self.assertRaises(ValueError):
            WeightedDaggerEpisodeSampler(
                from_idx,
                to_idx,
                action_source=action_source,
                is_dagger_per_frame=None,
                dagger_fraction=0.25,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
