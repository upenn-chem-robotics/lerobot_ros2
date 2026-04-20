"""Uniform-stride observation history for LeRobot diffusion policies.

Stock ``DiffusionPolicy.select_action`` only ever stitches *consecutive* past
frames (``obs_{t-n+1}, ..., obs_t``). For long-horizon history we want a
sparser, uniformly-spaced window, e.g. ``(obs_t, obs_{t-1s}, obs_{t-2s})`` at
whatever FPS the dataset runs at.

Training-time support lives in the ``lerobot_policy_strided_diffusion``
third-party plugin (see ``packages/lerobot_policy_strided_diffusion/``), which
registers ``StridedDiffusionConfig`` under the ``strided_diffusion`` policy
type. That config overrides ``observation_delta_indices`` so the canonical
``lerobot-train`` pipeline picks up the strided offsets via
``lerobot.datasets.factory.resolve_delta_timestamps`` with no other changes.

This module provides the inference-side glue:

- ``load_strided_config`` reads the canonical ``config.json`` emitted next to a
  trained checkpoint, detects ``type == "strided_diffusion"``, and returns the
  ``(fps, n_obs_steps, stride_seconds)`` triple needed by the runner.
- ``StridedHistoryRunner`` keeps its own strided ring buffer, populates
  ``policy._queues`` with the right frames, and drives
  ``policy.predict_action_chunk`` / action replay the same way stock
  ``DiffusionPolicy.select_action`` does.

Importing this module also triggers the plugin package's side-effect import so
that ``PreTrainedConfig.from_pretrained`` can dispatch ``type=strided_diffusion``
entries in ``config.json`` to ``StridedDiffusionConfig`` without requiring
callers to remember ``register_third_party_plugins()``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from lerobot.utils.constants import ACTION, OBS_IMAGES

try:
    import lerobot_policy_strided_diffusion  # noqa: F401  (registers StridedDiffusionConfig)
except ImportError:
    # The plugin isn't installed in this environment. Training/loading a
    # strided_diffusion checkpoint will fail downstream, but importing this
    # module should still succeed for callers that only need StridedHistoryRunner
    # on stock diffusion policies.
    pass


STRIDED_POLICY_TYPE = "strided_diffusion"


# ── Inference-side config discovery ─────────────────────────────────────


@dataclass
class StridedHistoryConfig:
    fps: int
    n_obs_steps: int
    stride_seconds: float = 1.0

    @property
    def stride_frames(self) -> int:
        frames = self.stride_seconds * self.fps
        if abs(round(frames) - frames) > 1e-6:
            raise ValueError(
                f"stride_seconds ({self.stride_seconds}) * fps ({self.fps}) "
                f"must be an integer; got {frames}."
            )
        return int(round(frames))

    @property
    def buffer_length(self) -> int:
        return self.stride_frames * (self.n_obs_steps - 1) + 1


def load_strided_config(policy_path: str | Path) -> Optional[StridedHistoryConfig]:
    """Look for a ``strided_diffusion`` policy config near ``policy_path``.

    Walks up the directory tree (matching ``deploy.py``'s sidecar discovery
    convention) for a ``config.json`` -- the canonical policy config emitted
    by ``PreTrainedConfig.save_pretrained`` -- and returns a lightweight
    ``StridedHistoryConfig`` iff that file says ``type == "strided_diffusion"``.
    Returns ``None`` for stock diffusion / ACT checkpoints.
    """
    base = Path(policy_path)
    if not base.exists():
        return None

    for parent in (base, *base.parents):
        candidate = parent / "config.json"
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("type") != STRIDED_POLICY_TYPE:
            # Keep walking: a stale sibling config.json (e.g. for the train
            # config or a preprocessor) shouldn't terminate the search early.
            # Only the policy's own config.json advertises a `type`, so a hit
            # is authoritative; a miss means this is the wrong file.
            continue
        try:
            return StridedHistoryConfig(
                fps=int(data["fps"]),
                n_obs_steps=int(data["n_obs_steps"]),
                stride_seconds=float(data.get("stride_seconds", 1.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


# ── Inference side: strided ring buffer around DiffusionPolicy ───────────


class StridedHistoryRunner:
    """Drive ``DiffusionPolicy`` with a strided-history observation window.

    ``DiffusionPolicy.predict_action_chunk`` stacks ``self._queues[k]``
    along ``dim=1`` and never looks at the ``batch`` argument for data
    (only for key filtering). So we can shortcut ``populate_queues`` and
    place exactly the strided frames we want into the queues before
    calling ``predict_action_chunk``.

    Call ``select_action`` once per environment frame (i.e. at the same
    ``fps`` the dataset was recorded at). The wrapper takes care of:

    - camera-key stacking into ``observation.images`` (mirrors stock
      ``select_action``),
    - warm-starting the strided buffer by replicating the first
      observation across every slot (matches lerobot's ``populate_queues``
      warm-start semantics),
    - action-chunk replay via an internal ``n_action_steps``-sized deque,
      using the same ``transpose(0, 1)`` / ``popleft`` idiom as stock.
    """

    def __init__(self, policy, fps: int, stride_seconds: float = 1.0):
        self.policy = policy
        self.fps = int(fps)
        self.stride_seconds = float(stride_seconds)

        n_obs = int(policy.config.n_obs_steps)
        if n_obs < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs}")
        stride_frames = self.stride_seconds * self.fps
        if abs(round(stride_frames) - stride_frames) > 1e-6:
            raise ValueError(
                f"stride_seconds * fps must be an integer; got "
                f"{self.stride_seconds} * {self.fps} = {stride_frames}."
            )

        self.n_obs: int = n_obs
        self.stride: int = int(round(stride_frames))
        self.buffer_length: int = self.stride * (self.n_obs - 1) + 1

        self._obs_hist: deque = deque(maxlen=self.buffer_length)
        self._act_queue: deque = deque(maxlen=policy.config.n_action_steps)

    # ----- lifecycle -------------------------------------------------

    def reset(self) -> None:
        """Mirror ``DiffusionPolicy.reset`` and clear the strided buffer."""
        self.policy.reset()
        self._obs_hist.clear()
        self._act_queue.clear()

    # ----- inference -------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = dict(batch)
        batch.pop(ACTION, None)

        if self.policy.config.image_features:
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.policy.config.image_features],
                dim=-4,
            )

        if len(self._obs_hist) == 0:
            for _ in range(self.buffer_length):
                self._obs_hist.append(batch)
        else:
            self._obs_hist.append(batch)

        if len(self._act_queue) == 0:
            idxs = [-1 - self.stride * k for k in reversed(range(self.n_obs))]
            frames = [self._obs_hist[i] for i in idxs]

            queue_keys = [k for k in self.policy._queues if k != ACTION]
            for key in queue_keys:
                q = self.policy._queues[key]
                q.clear()
                for frame in frames:
                    q.append(frame[key])

            actions = self.policy.predict_action_chunk(batch)
            self._act_queue.extend(actions.transpose(0, 1))

        return self._act_queue.popleft()

    # ----- introspection (handy for smoke tests) ---------------------

    def debug_queue_lengths(self) -> Dict[str, int]:
        return {k: len(q) for k, q in self.policy._queues.items()}

    def debug_buffer_length(self) -> int:
        return len(self._obs_hist)
