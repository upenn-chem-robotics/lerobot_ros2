"""DiffusionPolicy variant that conditions on a queue of prior commanded actions.

Architecturally this is stock ``DiffusionPolicy`` plus:

* a small ``Linear -> Mish -> Linear`` MLP that encodes the flattened
  ``(K * action_dim,)`` past-action vector into an ``action_history_embed_dim``
  embedding, and
* a wider U-Net whose FiLM ``global_cond_dim`` is grown by the same
  ``action_history_embed_dim`` so the past-action embedding can be appended to
  the global conditioning vector.

The training-time forward path splits ``batch["action"]`` (of length
``K + horizon``) back into ``(past_actions, target)``; the inference path
maintains an internal ring buffer of committed actions that is cleared by
:py:meth:`reset` and updated on every :py:meth:`select_action` call. Stock
``DiffusionPolicy`` behaviour is preserved exactly when ``n_action_history == 0``.

Past-action *dropout* (``action_history_dropout_prob > 0``) zeros the
past-action input on a fraction of training forward passes; this is the
de Haan et al. (2019) "causal confusion" mitigation. At inference the
past-action input is always populated from the queue.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.utils.constants import ACTION

from .configuration_action_history_diffusion import ActionHistoryDiffusionConfig


# Key used to carry the (B, n_action_history, action_dim) past-action tensor
# through the batch. Distinct from ``ACTION`` to keep target and conditioning
# separate inside ``_prepare_global_conditioning``.
ACTION_HISTORY = "action_history"
ACTION_HISTORY_IS_PAD = "action_history_is_pad"


def _stock_global_cond_dim(model: DiffusionModel) -> int:
    """Recompute the ``global_cond_dim`` that ``DiffusionModel.__init__`` would
    have passed to its U-Net for ``config``.

    Mirrors the logic at ``DiffusionModel.__init__`` (state + per-camera image
    features + optional env state, all multiplied by ``n_obs_steps``). We
    recompute here rather than caching a private attribute on the parent so
    the math stays in sync with upstream by inspection rather than by hidden
    state.
    """
    config = model.config
    per_step = config.robot_state_feature.shape[0]
    if config.image_features:
        num_images = len(config.image_features)
        if config.use_separate_rgb_encoder_per_camera:
            per_step += model.rgb_encoder[0].feature_dim * num_images
        else:
            per_step += model.rgb_encoder.feature_dim * num_images
    if config.env_state_feature:
        per_step += config.env_state_feature.shape[0]
    return per_step * config.n_obs_steps


class ActionHistoryDiffusionModel(DiffusionModel):
    """``DiffusionModel`` with an extra past-action conditioning branch."""

    def __init__(self, config: ActionHistoryDiffusionConfig):
        # Build the stock encoders + U-Net first; we'll overwrite the U-Net
        # below with a wider one whose ``global_cond_dim`` accounts for our
        # past-action embedding. The first U-Net is briefly allocated and then
        # discarded -- this is cheap (~tens of MB of params at most for this
        # config) and avoids duplicating the stock encoder-setup logic.
        super().__init__(config)

        if config.n_action_history <= 0:
            self.action_history_encoder = None
            return

        action_dim = config.action_feature.shape[0]
        history_input_dim = config.n_action_history * action_dim
        embed_dim = config.action_history_embed_dim

        self.action_history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )

        stock_cond_dim = _stock_global_cond_dim(self)
        new_global_cond_dim = stock_cond_dim + embed_dim
        # Replace the stock U-Net with one whose FiLM conditioning dim is
        # widened by the past-action embedding. ``compile_model`` is honoured
        # to match parent behaviour.
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=new_global_cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        cond = super()._prepare_global_conditioning(batch)
        if self.action_history_encoder is None:
            return cond

        if ACTION_HISTORY not in batch:
            # Defensive: if the policy wrapper didn't populate the key we feed
            # zeros (this is also what dropout looks like). Avoids silent
            # shape-mismatch crashes if a caller drives the model directly.
            batch_size = cond.shape[0]
            embed_dim = self.config.action_history_embed_dim
            zero_embed = torch.zeros(batch_size, embed_dim, device=cond.device, dtype=cond.dtype)
            return torch.cat([cond, zero_embed], dim=-1)

        history = batch[ACTION_HISTORY]
        # history is (B, K, action_dim). Flatten across (K, action_dim).
        history_flat = history.flatten(start_dim=1)

        # Training-time causal-confusion dropout: replace the entire
        # past-action vector with zeros on a per-sample basis. We zero the
        # input rather than the embedding so the dropout is independent of
        # the encoder's learned bias.
        p = self.config.action_history_dropout_prob
        if self.training and p > 0.0:
            keep_mask = (torch.rand(history_flat.shape[0], device=history_flat.device) >= p).to(
                history_flat.dtype
            )
            history_flat = history_flat * keep_mask.unsqueeze(-1)

        embed = self.action_history_encoder(history_flat)
        return torch.cat([cond, embed], dim=-1)


class ActionHistoryDiffusionPolicy(DiffusionPolicy):
    """``DiffusionPolicy`` that exposes a past-action conditioning queue."""

    config_class = ActionHistoryDiffusionConfig
    name = "action_history_diffusion"

    def __init__(self, config: ActionHistoryDiffusionConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        # Replace the stock DiffusionModel with our subclass that owns the
        # past-action encoder. We allocate twice (parent already built one)
        # but the saving in code clarity is worth a one-shot init cost.
        if config.n_action_history > 0:
            self.diffusion = ActionHistoryDiffusionModel(config)

        # Ring buffer of past committed actions, populated by select_action().
        # We keep at least one slot so the deque object always exists even when
        # the feature is disabled (simplifies branchless reset()).
        self._action_history_buffer: deque = deque(
            maxlen=max(config.n_action_history + max(config.n_obs_steps - 1, 0), 1)
        )

    # ── training ──────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.n_action_history > 0:
            batch = self._split_action_history_for_training(batch)
        return super().forward(batch)

    def _split_action_history_for_training(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Slice ``batch[ACTION]`` of length ``K + horizon`` back into
        ``(action_history, action)`` pieces with the shapes the parent class
        and the inner ``DiffusionModel`` expect.

        Also slices ``action_is_pad`` so the optional padding-loss masking
        keeps lining up with the target window.
        """
        K = self.config.n_action_history
        new_batch = dict(batch)

        full_action = batch[ACTION]
        # Sanity: caller should have configured action_delta_indices via our
        # config, which guarantees length K + horizon.
        expected_len = K + self.config.horizon
        if full_action.shape[1] != expected_len:
            raise ValueError(
                f"Expected batch[{ACTION!r}] to have time dim {expected_len} "
                f"(n_action_history + horizon), got {full_action.shape[1]}. "
                "Check that the dataset was loaded with this policy's config."
            )

        new_batch[ACTION_HISTORY] = full_action[:, :K]
        new_batch[ACTION] = full_action[:, K:]

        if "action_is_pad" in batch:
            pad = batch["action_is_pad"]
            new_batch[ACTION_HISTORY_IS_PAD] = pad[:, :K]
            new_batch["action_is_pad"] = pad[:, K:]

        return new_batch

    # ── inference ─────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        # Clear the past-action buffer; subsequent select_action calls warm-
        # start with zero-padded history until n_action_history actions have
        # been committed. Guarded with ``getattr`` because parent's __init__
        # calls reset() before our __init__ body has run.
        buf = getattr(self, "_action_history_buffer", None)
        if buf is not None:
            buf.clear()

    def _build_action_history_tensor(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Materialise a ``(B, n_action_history, action_dim)`` tensor from the
        internal buffer of committed actions.

        Alignment with training:

        * Training history corresponds to ``action_delta_indices`` in
          ``[1 - n_obs_steps - K, ..., -n_obs_steps]`` (the K indices strictly
          before the target chunk's first slot).
        * At inference, the action chunk we are about to predict starts at
          delta index ``1 - n_obs_steps`` relative to the current observation
          ``t``. So the training-aligned past actions are at absolute times
          ``t - n_obs_steps - K + 1, ..., t - n_obs_steps``. With
          ``n_obs_steps = 1`` (the default for this plugin) this is the K
          most recent committed actions, i.e. the tail of the buffer.

        We pad with zeros for the first few decisions (the same warm-start
        semantics as stock ``DiffusionPolicy``'s observation queue).
        """
        K = self.config.n_action_history
        action_dim = self.config.action_feature.shape[0]
        # Offset to drop the n_obs_steps - 1 most recent committed actions
        # (matching training-time delta indices). With n_obs_steps=1 this is 0.
        skip_recent = max(self.config.n_obs_steps - 1, 0)

        buf = list(self._action_history_buffer)
        # Buffer holds, oldest -> newest, all committed actions we've kept.
        # Drop the skip_recent newest entries first.
        if skip_recent > 0:
            buf = buf[:-skip_recent] if len(buf) >= skip_recent else []
        # Then take the K most recent (which are the K closest to t-skip_recent).
        usable = buf[-K:]

        # Each entry in buf is a (B, action_dim) tensor returned by
        # super().select_action / popped from self._queues[ACTION].
        if len(usable) < K:
            pad_count = K - len(usable)
            pad = torch.zeros((batch_size, action_dim), device=device, dtype=dtype)
            usable = [pad] * pad_count + usable

        # Stack to (B, K, action_dim). Each entry already has a batch dim.
        return torch.stack(usable, dim=1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # Parent stacks queue entries along dim=1. We replicate that, then
        # inject the past-action tensor before invoking the diffusion model.
        if self.config.n_action_history == 0:
            return super().predict_action_chunk(batch, noise=noise)

        # Replicate parent's batch construction (see DiffusionPolicy
        # .predict_action_chunk in modeling_diffusion.py).
        stock_batch = {
            k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues
        }
        # Pull dtype / device / batch size from whatever the queues delivered.
        ref = next(iter(stock_batch.values()))
        action_history = self._build_action_history_tensor(
            batch_size=ref.shape[0], device=ref.device, dtype=ref.dtype
        )
        stock_batch[ACTION_HISTORY] = action_history
        actions = self.diffusion.generate_actions(stock_batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        action = super().select_action(batch, noise=noise)
        if self.config.n_action_history > 0:
            # Stash a detached copy so we don't keep an autograd graph alive
            # across calls (no_grad already prevents this, but explicit detach
            # is cheap insurance for callers that wrap select_action).
            self._action_history_buffer.append(action.detach().clone())
        return action
