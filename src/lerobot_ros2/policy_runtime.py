#!/usr/bin/env python3
"""Shared runtime plumbing for the policy-driven CLIs (``deploy``, ``dagger``).

Both CLIs load the same checkpoints, recover the same training-time transforms,
and drive the same optional :class:`StridedHistoryRunner`. Keeping that here
means ``dagger`` no longer imports from ``deploy``: they are sibling commands,
not a library and its caller.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from lerobot_ros2.helper import (
    extract_home_joint_names,
    extract_home_positions,
    load_experiment_home_config,
)
from lerobot_ros2.preprocessing import (
    load_fullres_crops_from_train_config,
    load_per_camera_crops_from_train_config,
)
from lerobot_ros2.strided_history import StridedHistoryRunner, load_strided_config

try:
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Importing the plugin registers ``action_history_diffusion`` as a known policy
# type so checkpoints saved with that ``type`` field can be loaded below. The
# import is wrapped in try/except so the CLIs still work in environments where
# the plugin isn't installed (those just can't load action-history checkpoints,
# but stock diffusion / ACT keep working).
try:
    from lerobot_policy_action_history_diffusion import ActionHistoryDiffusionPolicy
    ACTION_HISTORY_AVAILABLE = True
except ImportError:
    ActionHistoryDiffusionPolicy = None  # type: ignore[assignment]
    ACTION_HISTORY_AVAILABLE = False


# ── ROS 2 command publisher ──────────────────────────────────────────────

class ROSActionPublisher:
    """Publishes JointState commands to an action topic."""

    def __init__(
        self,
        node: "Node",
        action_topic: str,
        joint_names: List[str],
        label: str = "",
    ) -> None:
        self._pub = node.create_publisher(JointStateMsg, action_topic, 10)
        self._joint_names = list(joint_names)
        logging.info(f"[{label}] Publishing actions to: {action_topic}")

    def send(self, joint_pos_7dof: np.ndarray) -> None:
        msg = JointStateMsg()
        msg.header.stamp.sec = 0  # immediate execution
        msg.header.stamp.nanosec = 0
        msg.name = self._joint_names
        msg.position = joint_pos_7dof.tolist()
        self._pub.publish(msg)


# ── Recovering training-time image transforms ────────────────────────────

def detect_training_resize(policy_path: str) -> Optional[Tuple[int, int]]:
    candidates = [
        Path(policy_path) / "train_config.json",
        Path(policy_path).parent / "train_config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                resize = cfg.get("dataset", {}).get("image_transforms", {}).get("resize")
                if resize and len(resize) == 2:
                    return tuple(resize)
            except (json.JSONDecodeError, KeyError):
                pass
    return None


def detect_per_camera_crops(policy_path: str) -> Dict[str, Dict[str, int]]:
    """Recover ``dataset.image_transforms.per_camera_crops`` from training.

    Mirrors :func:`detect_training_resize`: looks for ``train_config.json``
    next to ``policy_path`` (or in its parent dir) and returns the validated
    per-camera crop dictionary so deploy/dagger can apply the exact same
    front-camera crop the policy was trained with. Returns ``{}`` when the
    file is missing, malformed, or has no ``per_camera_crops`` block.
    """
    return load_per_camera_crops_from_train_config(policy_path)


def detect_fullres_crops(policy_path: str) -> Dict[str, Dict[str, int]]:
    """Recover the per-camera full-res crop ROIs baked into the dataset.

    Mirrors :func:`detect_per_camera_crops` but reads ``source_crop`` from the
    training dataset's ``meta/info.json`` (written by ``downsample.py``), so
    deploy can crop the raw camera frame to the same ROI before resizing.
    """
    return load_fullres_crops_from_train_config(policy_path)


def parse_fullres_crop_args(entries: Optional[list]) -> Dict[str, Dict[str, int]]:
    """Parse ``--fullres-crop NAME TOP LEFT HEIGHT WIDTH`` entries.

    NAME is the stripped camera name (e.g. ``front``); the returned mapping is
    keyed by the full ``observation.images.<name>`` feature key so it lines up
    with the deploy loop. Raises ``SystemExit`` on malformed specs.
    """
    crops: Dict[str, Dict[str, int]] = {}
    for entry in entries or []:
        if len(entry) != 5:
            raise SystemExit(
                f"--fullres-crop expects NAME TOP LEFT HEIGHT WIDTH; got {entry}"
            )
        name = str(entry[0]).strip()
        if not name:
            raise SystemExit(f"--fullres-crop has an empty NAME in {entry}")
        try:
            top, left, height, width = (int(v) for v in entry[1:])
        except ValueError:
            raise SystemExit(
                f"--fullres-crop TOP LEFT HEIGHT WIDTH must be integers; got {entry}"
            )
        if height <= 0 or width <= 0 or top < 0 or left < 0:
            raise SystemExit(f"--fullres-crop {name} has invalid bounds; got {entry}")
        key = name if name.startswith("observation.images.") else f"observation.images.{name}"
        crops[key] = {"top": top, "left": left, "height": height, "width": width}
    return crops


# ── Locating the configs that sit next to a checkpoint ───────────────────

def experiment_config_defines_home(path: Path) -> bool:
    """True if ``path`` is a YAML file carrying ``home_positions`` for an arm.

    Only a file with an actual home pose is usable here. Anything else (a
    plateau/filter sidecar, a bare camera config, ...) makes ``resolve_home``
    fall back to the *live* robot state, which silently turns "home" into
    "wherever the arms happened to be when deploy launched".
    """
    try:
        parsed = load_experiment_home_config(path)
    except (OSError, yaml.YAMLError):
        return False
    positions = extract_home_positions(parsed)
    names = extract_home_joint_names(parsed)
    return any(positions.get(arm) and names.get(arm) for arm in ("left", "right"))


def find_experiment_config_candidates(*search_roots: Optional[str]) -> List[Path]:
    """Walk up from each root looking for configs that define a home pose."""
    found: List[Path] = []
    for root in search_roots:
        if not root:
            continue
        base_path = Path(root)
        if not base_path.exists():
            continue
        for parent in (base_path, *base_path.parents):
            for candidate_name in ("experiment_config.yaml", "recording_config.yaml", "config.yaml"):
                candidate = parent / candidate_name
                if (
                    candidate.exists()
                    and candidate not in found
                    and experiment_config_defines_home(candidate)
                ):
                    found.append(candidate)
    return found


def resolve_experiment_config_path(policy_path: str, explicit_path: Optional[str]) -> Optional[Path]:
    """Locate the config that supplies the home pose.

    An explicit ``--experiment-config`` wins, but only if it actually defines a
    home pose; otherwise we fall back to searching next to that file (the
    dataset directory usually holds the real ``experiment_config.yaml`` a few
    levels up) and then next to the checkpoint.
    """
    if explicit_path:
        path = Path(explicit_path)
        if path.exists() and experiment_config_defines_home(path):
            return path
        if not path.exists():
            logging.warning("Requested experiment config %s does not exist", path)
        else:
            logging.warning(
                "Experiment config %s has no 'home_positions' for either arm; "
                "searching nearby directories for one that does",
                path,
            )
        fallback = find_experiment_config_candidates(explicit_path, policy_path)
        if fallback:
            logging.warning("Falling back to experiment config %s", fallback[0])
            return fallback[0]
        return path if path.exists() else None

    candidates = find_experiment_config_candidates(policy_path)
    return candidates[0] if candidates else None


def resolve_recording_config_path(policy_path: str, explicit_path: Optional[str]) -> Optional[Path]:
    """Resolve the ``recording_config.yaml`` saved alongside the training dataset.

    Mirrors ``resolve_experiment_config_path``: if the user passes an explicit
    path, use it; otherwise walk up from the policy directory looking for a
    sibling ``recording_config.yaml`` (the consolidated file record.py writes
    next to each dataset, carrying resolved camera + v4l2 snapshot info).
    """
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    base_path = Path(policy_path)
    if not base_path.exists():
        return None

    for parent in (base_path, *base_path.parents):
        candidate = parent / "recording_config.yaml"
        if candidate.exists():
            return candidate
    return None


# ── Loading the policy ───────────────────────────────────────────────────

def load_policy_type(policy_path: str, explicit_mode: str) -> str:
    if explicit_mode != "auto":
        return "diffusion" if explicit_mode == "dp" else "act"

    if not Path(policy_path).exists():
        raise FileNotFoundError(
            f"Policy path {policy_path!r} does not exist locally. "
            "Use --deploy-mode dp or --deploy-mode act when passing a HuggingFace repo id."
        )

    policy_cfg_path = Path(policy_path) / "config.json"
    if not policy_cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json under {policy_path!r}")

    policy_cfg = json.loads(policy_cfg_path.read_text())
    raw_type = policy_cfg.get("type", "act")
    # The strided_diffusion plugin (StridedDiffusionConfig subclass of
    # DiffusionConfig) is, from the deploy loop's perspective, a plain
    # DiffusionPolicy with a custom observation window -- the strided
    # behaviour is layered on via StridedHistoryRunner below. Keep the
    # downstream branching purely binary ("diffusion" vs "act") and let
    # load_strided_config pick up the stride parameters separately.
    if raw_type == "strided_diffusion":
        return "diffusion"
    # The action_history_diffusion plugin DOES change the model architecture
    # (extra MLP + wider U-Net), so we cannot collapse it into "diffusion"
    # the way we do for strided. Surface it as its own type and dispatch to
    # ActionHistoryDiffusionPolicy.from_pretrained below.
    if raw_type == "action_history_diffusion":
        return "action_history_diffusion"
    return raw_type


def resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA was asked for but isn't there."""
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return "cpu"
    return requested


def load_policy(policy_path: str, deploy_mode: str, device: str):
    """Load a checkpoint and return ``(policy, policy_type)`` ready for inference."""
    policy_type = load_policy_type(policy_path, deploy_mode)
    if policy_type == "action_history_diffusion":
        if not ACTION_HISTORY_AVAILABLE:
            raise RuntimeError(
                "Checkpoint declares type=action_history_diffusion but the "
                "lerobot_policy_action_history_diffusion plugin is not installed "
                "in this environment. Install it with "
                "`pip install -e packages/lerobot_policy_action_history_diffusion`."
            )
        policy = ActionHistoryDiffusionPolicy.from_pretrained(policy_path)
    elif policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(policy_path)
    else:
        policy = ACTPolicy.from_pretrained(policy_path)

    policy.config.device = device
    policy.to(device)
    policy.eval()
    # Both DiffusionPolicy and its action_history subclass need their queues
    # primed via reset() before the first select_action() call.
    if policy_type in ("diffusion", "action_history_diffusion") and hasattr(policy, "reset"):
        policy.reset()
    return policy, policy_type


def build_strided_runner(
    policy,
    policy_type: str,
    policy_path: str,
    hz: float,
):
    """Return ``(strided_cfg, strided_runner)``, both ``None`` when not strided.

    If the checkpoint was trained with a uniform-stride observation window
    (``policy.type == "strided_diffusion"``; see the
    lerobot_policy_strided_diffusion plugin), swap in a runner that keeps its
    own strided ring buffer and drives ``predict_action_chunk`` directly. The
    policy itself stays untouched.

    ``action_history_diffusion`` has its own queue layer inside the policy, so
    it does NOT use StridedHistoryRunner -- the two extensions are independent.
    """
    if policy_type != "diffusion":
        return None, None

    strided_cfg = load_strided_config(policy_path)
    if strided_cfg is None:
        return None, None

    if strided_cfg.n_obs_steps != policy.config.n_obs_steps:
        logging.warning(
            "strided_history.json says n_obs_steps=%d but the policy "
            "config reports %d; trusting the policy config.",
            strided_cfg.n_obs_steps,
            policy.config.n_obs_steps,
        )
        strided_cfg.n_obs_steps = int(policy.config.n_obs_steps)

    strided_runner = StridedHistoryRunner(
        policy,
        fps=int(strided_cfg.fps),
        stride_seconds=float(strided_cfg.stride_seconds),
    )

    expected_hz = int(strided_cfg.fps)
    if int(round(hz)) != expected_hz:
        logging.warning(
            "Strided history expects %d Hz inference (matches training fps); "
            "you requested --hz %.2f. The 1-second stride will be off unless "
            "--hz matches the dataset fps.",
            expected_hz,
            hz,
        )
    logging.info(
        "Strided history enabled: n_obs_steps=%d, stride=%.2fs, "
        "buffer_length=%d frames",
        strided_runner.n_obs,
        strided_runner.stride_seconds,
        strided_runner.buffer_length,
    )
    return strided_cfg, strided_runner


# ── Policy/robot shape reconciliation ────────────────────────────────────

def get_image_feature_keys(policy) -> List[str]:
    return [
        name
        for name, feature in policy.config.input_features.items()
        if feature.type == "VISUAL" and name.startswith("observation.images.")
    ]


def get_policy_visual_size(policy) -> Optional[Tuple[int, int]]:
    for feature in policy.config.input_features.values():
        if feature.type == "VISUAL" and len(feature.shape) == 3:
            return int(feature.shape[1]), int(feature.shape[2])
    return None


def resolve_arm_keys(
    arm_configs: Dict[str, object],
    arm_joint_names: Dict[str, List[str]],
    policy,
) -> List[str]:
    """Determine which arms to use based on the experiment config and policy state dim."""
    all_arms = [k for k in ("left", "right") if k in arm_configs]
    configured_arms = [k for k in all_arms if arm_joint_names.get(k)]

    state_feature = policy.config.robot_state_feature
    if state_feature is None:
        return configured_arms if configured_arms else all_arms

    expected_dim = state_feature.shape[0]

    if configured_arms:
        configured_dim = sum(len(arm_joint_names[k]) for k in configured_arms)
        if configured_dim == expected_dim:
            return configured_arms

    for arm in all_arms:
        names = arm_joint_names.get(arm, [])
        if len(names) == expected_dim:
            return [arm]

    return all_arms
