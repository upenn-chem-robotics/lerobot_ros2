"""LeRobot third-party plugin: DiffusionPolicy with past-action conditioning.

Importing this package registers ``action_history_diffusion`` as a policy type
on ``PreTrainedConfig`` so it can be selected via
``lerobot-train --policy.type=action_history_diffusion``.

Motivation
----------
The stock ``DiffusionPolicy`` conditions only on observations
(``observation.state`` plus images, optionally ``observation.environment_state``).
This plugin adds a separate, cheap conditioning branch: a small MLP that
encodes the last ``n_action_history`` *commanded* actions and concatenates the
resulting embedding into the FiLM global-conditioning vector fed to the 1D
U-Net.

Vision and proprioception conditioning are untouched, so the marginal compute
cost is a single ``Linear(K*action_dim, embed) -> Mish -> Linear(embed, embed)``
MLP per training/inference step plus an ``embed``-dim widening of the U-Net's
FiLM conditioning. Both are negligible compared to the ResNet vision backbone.

Past-action conditioning is well-suited to disambiguate phase ambiguity
("am I mid-pour or done?") at low compute cost. The known failure mode is
*causal confusion* (de Haan et al. 2019): expert actions are highly
autocorrelated and the policy may learn ``a_t ~ f(a_{t-1})`` while ignoring
the visual stream. ``action_history_dropout_prob`` (random zeroing of the
past-action input during training) provides a knob for mitigating this.

Example::

    lerobot-train \\
        --policy.type=action_history_diffusion \\
        --policy.n_obs_steps=1 \\
        --policy.n_action_history=8 \\
        --policy.horizon=64 \\
        --policy.n_action_steps=8 \\
        --dataset.repo_id=local/pour --dataset.root=data/pour \\
        --output_dir=outputs/dp_action_history

or, equivalently, point at a ``config.json`` with
``policy.type=action_history_diffusion`` and ``policy.n_action_history=K``.

Deploy-side
-----------
Inference-time queue management is fully encapsulated inside
``ActionHistoryDiffusionPolicy`` (it maintains its own past-action ring buffer
that gets cleared by ``reset()`` and updated by ``select_action()``).
``lerobot_ros2.cli.deploy`` detects ``type == "action_history_diffusion"`` and
loads the policy via ``ActionHistoryDiffusionPolicy.from_pretrained`` so that
the past-action encoder weights are restored.
"""

from .configuration_action_history_diffusion import ActionHistoryDiffusionConfig
from .modeling_action_history_diffusion import ActionHistoryDiffusionPolicy

__all__ = ["ActionHistoryDiffusionConfig", "ActionHistoryDiffusionPolicy"]
