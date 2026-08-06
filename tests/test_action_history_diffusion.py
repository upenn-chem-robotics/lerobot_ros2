"""Unit tests for the lerobot_policy_action_history_diffusion plugin.

Covers the four things that are easy to silently get wrong with this plugin
and that we therefore want guarded by tests:

1. ``action_delta_indices`` math (the only data-loader contract we expose).
2. Training-time batch split: ``batch[action]`` of length ``K + horizon``
   becomes ``batch[action_history]`` of length ``K`` plus a stock-shaped
   ``batch[action]`` of length ``horizon``.
3. End-to-end ``forward`` produces a finite scalar loss with both the
   ``K > 0`` path and the ``K = 0`` (parity-with-stock) path.
4. Inference queue alignment: after enough ``select_action`` calls,
   ``_build_action_history_tensor`` returns the K most recently committed
   actions (for ``n_obs_steps = 1``), and zero-pads from the left during
   warm-up.

The tests build the smallest possible diffusion config that satisfies
``DiffusionConfig.__post_init__`` validation (horizon divisible by the U-Net
downsampling factor, at least one image feature) so they run on CPU in a
second or two.

Run::

    /opt/conda/envs/lerobot/bin/python -m pytest tests/test_action_history_diffusion.py -q
"""

from __future__ import annotations

import unittest

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_action_history_diffusion import (
    ActionHistoryDiffusionConfig,
    ActionHistoryDiffusionPolicy,
)
from lerobot_policy_action_history_diffusion.modeling_action_history_diffusion import (
    ACTION_HISTORY,
)


def _make_config(
    *,
    n_obs_steps: int = 1,
    horizon: int = 16,
    n_action_history: int = 0,
    action_history_dropout_prob: float = 0.0,
) -> ActionHistoryDiffusionConfig:
    """Minimal config that survives ``DiffusionConfig.__post_init__``.

    ``horizon=16`` and the default ``down_dims=(512, 1024, 2048)`` together
    satisfy the ``horizon % 2**len(down_dims) == 0`` check. We shrink
    ``down_dims`` and turn off the rgb encoder to keep the U-Net small.
    """
    cfg = ActionHistoryDiffusionConfig(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=min(8, horizon - n_obs_steps),
        n_action_history=n_action_history,
        action_history_embed_dim=16,
        action_history_dropout_prob=action_history_dropout_prob,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        diffusion_step_embed_dim=16,
        spatial_softmax_num_keypoints=4,
        crop_shape=None,
        resize_shape=None,
        vision_backbone="resnet18",
        use_group_norm=True,
        # drop_n_last_frames=horizon - n_action_steps - n_obs_steps + 1 (stock formula).
        drop_n_last_frames=horizon - min(8, horizon - n_obs_steps) - n_obs_steps + 1,
        num_train_timesteps=10,
        num_inference_steps=2,
        compile_model=False,
    )
    # Mirror the pour task minus cameras (we'll add one image feature below to
    # satisfy validate_features()).
    cfg.input_features = {
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return cfg


def _make_batch(cfg: ActionHistoryDiffusionConfig, batch_size: int = 2):
    """Build a synthetic batch with the shapes the data loader would produce."""
    state = torch.zeros(batch_size, cfg.n_obs_steps, 7)
    img = torch.zeros(batch_size, cfg.n_obs_steps, 3, 32, 32)
    # Action length comes from action_delta_indices (which we are also testing).
    action_len = len(cfg.action_delta_indices)
    action = torch.zeros(batch_size, action_len, 7)
    action_is_pad = torch.zeros(batch_size, action_len, dtype=torch.bool)
    return {
        "observation.state": state,
        "observation.images.cam": img,
        "action": action,
        "action_is_pad": action_is_pad,
    }


class TestActionDeltaIndices(unittest.TestCase):
    def test_disabled_matches_stock(self):
        # K=0 must be the stock window list(range(1-n_obs_steps, 1-n_obs_steps+horizon))
        for n_obs in (1, 2, 3):
            cfg = _make_config(n_obs_steps=n_obs, horizon=16, n_action_history=0)
            expected = list(range(1 - n_obs, 1 - n_obs + 16))
            self.assertEqual(cfg.action_delta_indices, expected)

    def test_enabled_prepends_strict_past(self):
        # K=4, n_obs_steps=1 -> history [-4,-3,-2,-1] + target [0..15].
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=4)
        self.assertEqual(
            cfg.action_delta_indices,
            [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )

    def test_enabled_n_obs_steps_2_no_overlap(self):
        # K=3, n_obs_steps=2. Stock target starts at -1, so history must be
        # strictly before -1 -> [-4,-3,-2] + target [-1..14].
        cfg = _make_config(n_obs_steps=2, horizon=16, n_action_history=3)
        self.assertEqual(
            cfg.action_delta_indices,
            [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        )
        # Overlap-freeness: max(history) < min(target) - 1 + 1 = min(target).
        history = cfg.action_delta_indices[: cfg.n_action_history]
        target = cfg.action_delta_indices[cfg.n_action_history :]
        self.assertLess(max(history), min(target))


class TestConfigValidation(unittest.TestCase):
    def test_negative_history_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(n_action_history=-1)

    def test_dropout_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(n_action_history=4, action_history_dropout_prob=1.5)

    def test_zero_embed_dim_rejected(self):
        with self.assertRaises(ValueError):
            ActionHistoryDiffusionConfig(
                n_action_history=4, action_history_embed_dim=0
            )


class TestBatchSplit(unittest.TestCase):
    def test_split_shapes(self):
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=5)
        policy = ActionHistoryDiffusionPolicy(cfg)
        batch = _make_batch(cfg, batch_size=3)
        out = policy._split_action_history_for_training(batch)
        self.assertEqual(out[ACTION_HISTORY].shape, (3, 5, 7))
        self.assertEqual(out["action"].shape, (3, 16, 7))
        self.assertEqual(out["action_history_is_pad"].shape, (3, 5))
        self.assertEqual(out["action_is_pad"].shape, (3, 16))

    def test_split_rejects_wrong_length(self):
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=5)
        policy = ActionHistoryDiffusionPolicy(cfg)
        bad = {"action": torch.zeros(2, 16, 7)}  # missing the 5 history slots
        with self.assertRaises(ValueError):
            policy._split_action_history_for_training(bad)


class TestForwardLoss(unittest.TestCase):
    def test_forward_with_history(self):
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=4)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.train()
        batch = _make_batch(cfg, batch_size=2)
        loss, _ = policy.forward(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_disabled_parity(self):
        # K=0 should produce a stock-shaped loss path, exercising the
        # branch where ActionHistoryDiffusionModel is NOT swapped in.
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=0)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.train()
        batch = _make_batch(cfg, batch_size=2)
        loss, _ = policy.forward(batch)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_with_dropout(self):
        # action_history_dropout_prob=1.0 zeros every sample; should still
        # produce finite loss and not raise.
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=4,
                           action_history_dropout_prob=1.0)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.train()
        batch = _make_batch(cfg, batch_size=2)
        loss, _ = policy.forward(batch)
        self.assertTrue(torch.isfinite(loss))


class TestInferenceQueue(unittest.TestCase):
    def test_warm_start_zero_pads(self):
        # Before any select_action call, the history tensor should be all zeros.
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=3)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.eval()
        history = policy._build_action_history_tensor(
            batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(history.shape, (1, 3, 7))
        self.assertTrue(torch.allclose(history, torch.zeros_like(history)))

    def test_tail_alignment_after_commits(self):
        # After committing actions a0, a1, ..., aN-1, the K most recent
        # (for n_obs_steps=1) should appear in order in the history tensor.
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=3)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.eval()
        committed = [torch.full((1, 7), float(i)) for i in range(5)]
        for a in committed:
            policy._action_history_buffer.append(a)
        history = policy._build_action_history_tensor(
            batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        # Expect [a2, a3, a4] (last K=3 of [a0..a4]).
        self.assertTrue(torch.allclose(history[0, 0], torch.full((7,), 2.0)))
        self.assertTrue(torch.allclose(history[0, 1], torch.full((7,), 3.0)))
        self.assertTrue(torch.allclose(history[0, 2], torch.full((7,), 4.0)))

    def test_partial_warm_start(self):
        # 2 commits + K=4 should produce 2 leading zero rows then a0, a1.
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=4)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.eval()
        a0 = torch.full((1, 7), 10.0)
        a1 = torch.full((1, 7), 20.0)
        policy._action_history_buffer.append(a0)
        policy._action_history_buffer.append(a1)
        history = policy._build_action_history_tensor(
            batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(history.shape, (1, 4, 7))
        self.assertTrue(torch.allclose(history[0, 0], torch.zeros(7)))
        self.assertTrue(torch.allclose(history[0, 1], torch.zeros(7)))
        self.assertTrue(torch.allclose(history[0, 2], torch.full((7,), 10.0)))
        self.assertTrue(torch.allclose(history[0, 3], torch.full((7,), 20.0)))

    def test_reset_clears_buffer(self):
        cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=3)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.eval()
        policy._action_history_buffer.append(torch.full((1, 7), 5.0))
        self.assertGreater(len(policy._action_history_buffer), 0)
        policy.reset()
        self.assertEqual(len(policy._action_history_buffer), 0)

    def test_n_obs_steps_2_skip_recent(self):
        # With n_obs_steps=2, alignment requires dropping the most recent
        # committed action and using the K before. Buffer holds [a0..a4];
        # K=3; expected history = [a1, a2, a3].
        cfg = _make_config(n_obs_steps=2, horizon=16, n_action_history=3)
        policy = ActionHistoryDiffusionPolicy(cfg)
        policy.eval()
        for i in range(5):
            policy._action_history_buffer.append(torch.full((1, 7), float(i)))
        history = policy._build_action_history_tensor(
            batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertTrue(torch.allclose(history[0, 0], torch.full((7,), 1.0)))
        self.assertTrue(torch.allclose(history[0, 1], torch.full((7,), 2.0)))
        self.assertTrue(torch.allclose(history[0, 2], torch.full((7,), 3.0)))


class TestParameterCount(unittest.TestCase):
    """The whole pitch of this plugin is 'cheap'. Verify the extra params
    are small compared to the stock diffusion model."""

    def test_extra_params_are_small(self):
        stock_cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=0)
        stock_policy = ActionHistoryDiffusionPolicy(stock_cfg)
        stock_params = sum(p.numel() for p in stock_policy.parameters())

        hist_cfg = _make_config(n_obs_steps=1, horizon=16, n_action_history=8)
        hist_policy = ActionHistoryDiffusionPolicy(hist_cfg)
        hist_params = sum(p.numel() for p in hist_policy.parameters())

        extra = hist_params - stock_params
        # MLP: Linear(8*7, 16) + Linear(16, 16) ~ 16*(56+1) + 16*(16+1) = 1184
        # plus the U-Net widening (FiLM cond_dim grows by 16 across many
        # residual blocks). Still tiny compared to the rgb encoder.
        # Just sanity-check it's positive and < 5% of stock.
        self.assertGreater(extra, 0)
        self.assertLess(extra / stock_params, 0.05)


if __name__ == "__main__":
    unittest.main()
