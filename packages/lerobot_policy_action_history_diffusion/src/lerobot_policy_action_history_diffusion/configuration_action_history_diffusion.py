"""Configuration for DiffusionPolicy with past-action conditioning.

Adds three fields on top of stock ``DiffusionConfig``:

- ``n_action_history``: number of prior commanded actions fed as conditioning.
  ``0`` (the default) disables the feature, in which case this policy is
  numerically identical to stock ``DiffusionPolicy``.
- ``action_history_embed_dim``: width of the MLP that encodes the flattened
  past-action vector into a per-step embedding concatenated onto the FiLM
  global conditioning vector.
- ``action_history_dropout_prob``: probability of zeroing the past-action
  input during a training forward pass (causal-confusion mitigation, cf.
  de Haan et al. 2019). At inference the past-action input is always used.

The only data-loader change is :pyattr:`action_delta_indices`, which prepends
``n_action_history`` extra past offsets to the stock target window. The
resulting ``batch[action]`` tensor has length ``n_action_history + horizon``;
:class:`ActionHistoryDiffusionPolicy` splits it back into a past-action
conditioning slice and the stock-shaped denoising target before invoking the
inherited training / generation code.

Delta-index semantics
---------------------
History indices are placed *strictly before* the stock target window. With
``base_start = 1 - n_obs_steps`` (i.e. the first stock target index)::

    history = [base_start - K, base_start - K + 1, ..., base_start - 1]
    target  = [base_start,     base_start + 1,     ..., base_start + horizon - 1]

For ``n_obs_steps = 1`` (the most common case here, and the one used by the
pour task), this collapses to the natural definition::

    history = [-K, ..., -1]   # last K committed actions
    target  = [ 0, ..., H-1]  # action chunk to predict

For ``n_obs_steps > 1`` the history is offset by ``n_obs_steps - 1`` so it
never overlaps the target chunk; see the inference-time queue indexing in
:class:`ActionHistoryDiffusionPolicy` for the matching adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("action_history_diffusion")
@dataclass
class ActionHistoryDiffusionConfig(DiffusionConfig):
    n_action_history: int = 0
    action_history_embed_dim: int = 128
    action_history_dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_history < 0:
            raise ValueError(
                f"`n_action_history` must be >= 0, got {self.n_action_history}."
            )
        if self.action_history_embed_dim <= 0:
            raise ValueError(
                f"`action_history_embed_dim` must be > 0, got {self.action_history_embed_dim}."
            )
        if not (0.0 <= self.action_history_dropout_prob <= 1.0):
            raise ValueError(
                "`action_history_dropout_prob` must be in [0, 1], got "
                f"{self.action_history_dropout_prob}."
            )

    @property
    def action_delta_indices(self) -> list:
        # Stock target: list(range(1 - n_obs_steps, 1 - n_obs_steps + horizon))
        base = super().action_delta_indices
        if self.n_action_history == 0:
            return base
        base_start = base[0]
        history = list(range(base_start - self.n_action_history, base_start))
        return history + base
