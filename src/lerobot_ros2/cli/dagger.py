#!/usr/bin/env python3
"""
DAgger-style correction recorder for UR3e + GELLO — LeRobot dataset format. 

Combines :mod:`lerobot_ros2.cli.deploy` (policy inference) and
:mod:`lerobot_ros2.cli.record` (pedal/GELLO handling, LeRobot dataset writes)
to collect RaC/IWR-style human-correction episodes:

1. Pedal 1 (1st press) — start policy rollout. Policy commands are published
   to the arm action topic. Nothing is written to disk yet; observations and
   the commanded policy action flow into a rolling pre-intervention buffer
   sized to cover Diffusion Policy's ``n_obs_steps × stride`` history window.
2. Pedal 1 (2nd press) — freeze. The policy publisher is silenced, inference
   stops, and the prebuffer stops accepting new frames. GELLO is still IDLE so
   the robot holds its last commanded pose. This gives the operator idle time 
   to position themselves for handover without polluting the saved episode
   with "frozen" states.
3. Pedal 1 (3rd press) — human takeover. GELLO ``control_mode`` transitions
   IDLE -> NORMAL; once GELLO resumes publishing (``/transition_ready``), the
   pre-intervention buffer (captured during step 1 only) is flushed into the
   episode (tagged ``action_source = 0``) and live GELLO frames are appended
   from here on (tagged ``action_source = 1``).
4. Pedal 1 (4th press) — end episode. ``dataset.save_episode()`` persists the
   pre-buffer + correction; GELLO goes IDLE again. The robot holds its last
   pose until the operator presses the middle pedal to home/reset, then the
   cycle repeats.

Pedal 2 (middle, KEY_B) — discard during an active episode; home + reset
when idle (same semantics as ``record.py``).
Pedal 3 (right, KEY_C) — cycle GELLO rotate mode (NORMAL → CW → CCW).

Controls (keyboard):
    s — cycle phase (IDLE → POLICY_ROLLOUT → FROZEN → TELEOP_CORRECTION → save → IDLE)
    d — discard / reset (discard while recording, reset while idle)
    r — reset while idle
    m — cycle GELLO mode (NORMAL → CW → CCW)
    q — quit and finalize dataset

Usage:
    lerobot-ros-dagger --name pick_place --task "pick up the block" \\
        --policy outputs/dp_pick_place/checkpoints/last/pretrained_model
    lerobot-ros-dagger --name ... --task "..." --policy ... --visualize \\
        --recording-control keyboard,pedal
        
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import v2 as transforms_v2

os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)

from lerobot_ros2.config_paths import resolve_config_path
from lerobot_ros2.hub_sync import try_sync_to_hub
from lerobot_ros2.helper import (
    EVDEV_AVAILABLE,
    ArmState,
    CameraReader,
    CameraStabilizer,
    _slugify_camera_name,
    FootPedalThread,
    PEDAL_DEFAULT_DEVICE,
    RobotHomeSender,
    ROSArmStateListener,
    WrapJointManager,
    build_recording_snapshot,
    load_arm_configs,
    load_camera_configs,
    load_config,
    load_experiment_home_config,
    open_configured_cameras,
    read_camera_overrides,
    resolve_home,
    resolve_pedal_evdev_paths,
    resolve_unwrap_config,
    resolve_wrap_joints,
    run_cbreak_keyboard_loop,
    save_experiment_home_config,
)
from lerobot_ros2.visualizer import LivePreview

# Reuse the GELLO control helpers and shared recording scaffolding from
# record.py: they already implement exactly the handover surface we need
# (control_mode IDLE <-> NORMAL + transition_ready handshake, reset service
# client, stats/resume, and the UserCommandExecutor single-in-flight mutex).
from lerobot_ros2.cli.record import (
    GelloControlModeClient,
    GelloTransitionReadyClient,
    RecordingStats,
    ResetRequestClient,
    ResetServiceClient,
    UserCommandExecutor,
    resolve_control_mode_services,
    resolve_reset_services,
    resolve_transition_ready_services,
)

# Reuse the policy-loader plumbing from deploy.py so ACT / (strided_)diffusion
# checkpoints load the same way and ``StridedHistoryRunner`` is driven
# identically to the standalone deployer.
from lerobot_ros2.cli.deploy import (
    ROSActionPublisher,
    _image_feature_keys,
    _load_policy_type,
    _policy_visual_size,
    _resolve_arm_keys,
    detect_fullres_crops,
    detect_per_camera_crops,
    detect_training_resize,
    parse_fullres_crop_args,
    resolve_experiment_config_path,
    resolve_recording_config_path,
)
from lerobot_ros2.preprocessing import apply_fullres_crop, apply_per_camera_crop

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg
    from std_srvs.srv import Trigger as TriggerSrv
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from lerobot_ros2.strided_history import StridedHistoryRunner, load_strided_config

# Importing the plugin registers ``action_history_diffusion`` as a known policy
# type so checkpoints saved with that ``type`` field can be loaded below. The
# import is wrapped in try/except so dagger.py still works in environments
# where the plugin isn't installed (those just can't load action-history
# checkpoints, but stock diffusion / ACT keep working). Mirrors deploy.py.
try:
    from lerobot_policy_action_history_diffusion import (
        ActionHistoryDiffusionPolicy,
    )
    _ACTION_HISTORY_AVAILABLE = True
except ImportError:
    ActionHistoryDiffusionPolicy = None  # type: ignore[assignment]
    _ACTION_HISTORY_AVAILABLE = False


# ── Phase state ───────────────────────────────────────────────────────────

ACTION_SOURCE_POLICY = 0
ACTION_SOURCE_HUMAN = 1


class Phase(IntEnum):
    IDLE = 0
    POLICY_ROLLOUT = 1
    FROZEN = 2
    TELEOP_CORRECTION = 3
    SAVING = 4


class DaggerState:
    """DAgger recorder state machine + transient UI metadata.

    Runs a simple 3-step pedal cycle IDLE -> POLICY -> TELEOP -> (save) IDLE
    plus a SAVING guard so the main loop never flushes frames while the
    dataset is mid-save_episode. All mutators are lock-guarded so the
    keyboard/pedal threads and the @hz loop thread can call them freely.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase: Phase = Phase.IDLE
        self.episode_idx: int = 0
        self._save_requested: bool = False
        self._save_discard: bool = False
        self._last_toggle_time = 0.0
        self._is_resetting = False
        self._capture_paused = False
        self._transient_label: Optional[str] = None
        self._transient_until: float = 0.0
        self.policy_publish_enabled = threading.Event()

    # ----- phase queries -------------------------------------------------

    @property
    def phase(self) -> Phase:
        with self._lock:
            return self._phase

    @property
    def is_policy_rollout(self) -> bool:
        return self.phase == Phase.POLICY_ROLLOUT

    @property
    def is_frozen(self) -> bool:
        return self.phase == Phase.FROZEN

    @property
    def is_teleop(self) -> bool:
        return self.phase == Phase.TELEOP_CORRECTION

    @property
    def is_saving(self) -> bool:
        return self.phase == Phase.SAVING

    @property
    def is_idle(self) -> bool:
        return self.phase == Phase.IDLE

    @property
    def is_recording(self) -> bool:
        """True whenever an episode is in-progress (policy, frozen, or teleop phase)."""
        with self._lock:
            return self._phase in (
                Phase.POLICY_ROLLOUT,
                Phase.FROZEN,
                Phase.TELEOP_CORRECTION,
            )

    @property
    def is_resetting(self) -> bool:
        with self._lock:
            return self._is_resetting

    def set_resetting(self, value: bool) -> None:
        with self._lock:
            self._is_resetting = value

    @property
    def is_capture_paused(self) -> bool:
        with self._lock:
            return self._capture_paused

    def set_capture_paused(self, value: bool) -> None:
        with self._lock:
            self._capture_paused = value

    # ----- phase transitions --------------------------------------------

    def _check_debounce(self) -> bool:
        now = time.monotonic()
        if now - self._last_toggle_time < 0.4:
            return False
        self._last_toggle_time = now
        return True

    def begin_policy_rollout(self, label: str = "") -> bool:
        with self._lock:
            if self._phase != Phase.IDLE:
                logging.warning("[%s] Cannot start policy: phase=%s", label, self._phase.name)
                return False
            if not self._check_debounce():
                return False
            self._phase = Phase.POLICY_ROLLOUT
            self._transient_label = None
            self._transient_until = 0.0
        self.policy_publish_enabled.set()
        logging.info("[%s] ▶ POLICY ROLLOUT started — episode %d", label, self.episode_idx)
        return True

    def begin_frozen(self, label: str = "") -> bool:
        """Pedal 1 press 2: silence the publisher and freeze the scene.

        The robot holds its last commanded pose (no GELLO, no policy publish,
        no prebuffer pushes). Caller is responsible for leaving GELLO in IDLE.
        """
        with self._lock:
            if self._phase != Phase.POLICY_ROLLOUT:
                logging.warning("[%s] Cannot freeze: phase=%s", label, self._phase.name)
                return False
            if not self._check_debounce():
                return False
            self._phase = Phase.FROZEN
        self.policy_publish_enabled.clear()
        logging.info("[%s] ❄ FROZEN — episode %d (pedal 1 → handover)", label, self.episode_idx)
        return True

    def begin_teleop(self, label: str = "") -> bool:
        with self._lock:
            if self._phase != Phase.FROZEN:
                logging.warning("[%s] Cannot take over: phase=%s", label, self._phase.name)
                return False
            if not self._check_debounce():
                return False
            self._phase = Phase.TELEOP_CORRECTION
        self.policy_publish_enabled.clear()
        logging.info("[%s] ✋ TELEOP handover — episode %d", label, self.episode_idx)
        return True

    def end_episode(self, label: str = "", save: bool = True) -> bool:
        with self._lock:
            if self._phase not in (
                Phase.POLICY_ROLLOUT,
                Phase.FROZEN,
                Phase.TELEOP_CORRECTION,
            ):
                logging.warning("[%s] Cannot end episode: phase=%s", label, self._phase.name)
                return False
            if not self._check_debounce():
                return False
            self._phase = Phase.SAVING
            self._save_requested = True
            self._save_discard = not save
        self.policy_publish_enabled.clear()
        if save:
            logging.info("[%s] ■ Episode %d ending — save requested", label, self.episode_idx)
        else:
            logging.info("[%s] X Episode %d ending — discard requested", label, self.episode_idx)
        return True

    def discard(self, label: str = "") -> bool:
        """Abort in-progress episode (from POLICY or TELEOP phase)."""
        return self.end_episode(label=label, save=False)

    def take_save_request(self) -> Tuple[bool, bool]:
        """Called by the recording thread: returns (save_flag, discard_flag).

        After draining, the state machine returns to IDLE and increments
        ``episode_idx`` on a successful save.
        """
        with self._lock:
            if not self._save_requested:
                return False, False
            discard = self._save_discard
            self._save_requested = False
            self._save_discard = False
            self._phase = Phase.IDLE
            if not discard:
                self.episode_idx += 1
            return True, discard

    def force_idle(self) -> None:
        with self._lock:
            self._phase = Phase.IDLE
            self._save_requested = False
            self._save_discard = False
        self.policy_publish_enabled.clear()

    # ----- transient UI label -------------------------------------------

    def set_transient_label(self, label: str, duration_s: float = 0.9) -> None:
        with self._lock:
            self._transient_label = label
            self._transient_until = time.monotonic() + max(duration_s, 0.0)

    def get_transient_label(self) -> Optional[str]:
        with self._lock:
            if self._transient_label is None:
                return None
            if time.monotonic() > self._transient_until:
                self._transient_label = None
                self._transient_until = 0.0
                return None
            return self._transient_label


# ── Pre-intervention ring buffer ──────────────────────────────────────────


class PreInterventionBuffer:
    """Fixed-size ring of (frame_dict, action_ndarray) tuples.

    We keep the last ``max_frames`` frames while the policy is running so
    that when the human intervenes we can flush a small trailing window
    into the dataset, tagged ``action_source = 0``. This preserves enough
    history for Diffusion Policy's ``n_obs_steps × stride`` observation
    window to be valid at the takeover boundary.
    """

    def __init__(self, max_frames: int) -> None:
        self.max_frames: int = max(int(max_frames), 1)
        self._buf: Deque[dict] = deque(maxlen=self.max_frames)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, frame: dict) -> None:
        self._buf.append(frame)

    def drain(self) -> List[dict]:
        out = list(self._buf)
        self._buf.clear()
        return out


def _resolve_prebuffer_frames(
    policy,
    strided_cfg,
    hz: float,
    override_seconds: Optional[float],
    margin_frames: int,
) -> int:
    """Determine the pre-intervention buffer size.

    When ``override_seconds`` is passed, ``buffer = ceil(override_seconds * hz) + margin``.
    Otherwise ``buffer = n_obs_steps * stride_frames + margin``, where the
    stride comes from ``strided_history.json`` if present (matches
    ``StridedHistoryRunner``) and defaults to 1 for stock diffusion/ACT.
    """
    if override_seconds is not None:
        return max(int(math.ceil(float(override_seconds) * float(hz))) + int(margin_frames), 1)

    n_obs = int(getattr(policy.config, "n_obs_steps", 1))
    if strided_cfg is not None:
        stride_frames = int(strided_cfg.stride_frames)
    else:
        stride_frames = 1
    return max(n_obs * stride_frames + int(margin_frames), 1)


# ── Dataset features ──────────────────────────────────────────────────────


def build_features(
    arm_labels: List[str],
    arm_joint_names: Dict[str, List[str]],
    cameras: List[CameraReader],
    resolution: Optional[Tuple[int, int]] = None,
) -> dict:
    joint_names: List[str] = []
    for arm in arm_labels:
        names = arm_joint_names.get(arm) or []
        if not names:
            raise ValueError(f"Missing joint names for arm {arm!r}")
        joint_names += names

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        # Per-frame label: 0 = policy-driven, 1 = human teleop. Used at
        # training time to mask loss to the human segment while keeping the
        # policy prefix available as Diffusion Policy observation context.
        "action_source": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["action_source"],
        },
    }
    for cam in sorted(cameras, key=lambda camera: camera.index):
        h = resolution[0] if resolution else cam.height
        w = resolution[1] if resolution else cam.width
        features[cam.feature_key] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# ── Argument parsing ──────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record DAgger correction episodes (policy rollout + human teleop)."
    )
    parser.add_argument("--name", required=True,
                        help="Dataset name. Data saved to data/<name>_dagger/.")
    parser.add_argument("--task", required=True,
                        help='Task description, e.g. "pick up the block".')
    parser.add_argument("--policy", required=True,
                        help="Path to pretrained model dir (or HF repo id).")
    parser.add_argument(
        "--deploy-mode",
        type=str,
        default="auto",
        choices=("auto", "act", "dp"),
        help=(
            "Force a deploy path. Use dp for DiffusionPolicy (and "
            "action_history_diffusion) checkpoints. Default 'auto' reads "
            "config.json and dispatches automatically."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--left", action="store_true", help="Left arm only.")
    group.add_argument("--right", action="store_true", help="Right arm only.")
    parser.add_argument("--hz", type=float,
                        default=None,
                        help="Recording/inference frequency in Hz. "
                             "Defaults to recording.hz from --config.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda).")
    parser.add_argument(
        "--prebuffer-seconds",
        type=float,
        default=None,
        help=(
            "Size of the rolling pre-intervention buffer, in seconds. "
            "Default: auto from the policy (n_obs_steps × stride)."
        ),
    )
    parser.add_argument(
        "--prebuffer-margin-frames",
        type=int,
        default=5,
        help="Extra frames added on top of the auto buffer size (default: 5).",
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds with phase overlay.")
    parser.add_argument(
        "--recording-control",
        type=str,
        default="keyboard",
        help="Comma-separated list of: keyboard, pedal.",
    )
    parser.add_argument(
        "--pedal-device",
        type=str,
        default=PEDAL_DEFAULT_DEVICE,
        help="evdev device path for the foot pedal (default: %(default)s).",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Record at HxW resolution. Default: native camera resolution.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the named camera/teleop config. Defaults to $GELLO_CONFIG.")
    parser.add_argument("--experiment-config", type=str, default=None,
                        help="Path to experiment_config.yaml saved with the dataset.")
    parser.add_argument(
        "--camera-settings",
        type=str,
        default=None,
        help=(
            "Path to recording_config.yaml saved alongside the training "
            "dataset. Auto-discovered near --policy by default; pass '' to force gello.yaml."
        ),
    )
    parser.add_argument(
        "--fullres-crop",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "TOP", "LEFT", "HEIGHT", "WIDTH"),
        help=(
            "Override the per-camera full-res crop ROI applied to the raw frame "
            "before feeding the policy (matches training). Repeatable. When "
            "omitted, auto-detected from the training dataset's info.json."
        ),
    )
    orient_group = parser.add_mutually_exclusive_group()
    orient_group.add_argument(
        "--stabilize-orientation",
        action="store_true",
        help="Enable 180° orientation stabilization for every camera.",
    )
    orient_group.add_argument(
        "--no-stabilize-orientation",
        action="store_true",
        help="Disable orientation stabilization for every camera.",
    )
    parser.add_argument(
        "--orientation-mae-delta-threshold",
        type=float,
        default=None,
        metavar="MAE",
        help="Override MAE gap (normal vs flipped) for all cameras.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Mirror the dagger dataset to Hugging Face when done "
             "(default: no upload; or set LEROBOT_HF_PUSH=1).",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not ROS_AVAILABLE:
        sys.exit("rclpy not found. Source your ROS 2 workspace first:\n  source /opt/ros/humble/setup.bash")
    if not LEROBOT_AVAILABLE:
        sys.exit("LeRobot not found. Install: pip install -e lerobot/")

    args = parse_args()

    # ── Config resolution (cameras + arms) ──────────────────────────────
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)

    # --hz defaults to recording.hz from the config when not supplied on CLI.
    # Done here (rather than inside parse_args) because parse_args runs before
    # --config has been resolved.
    if args.hz is None:
        args.hz = float(cfg.get("recording", {}).get("hz", 10.0))

    camera_settings_path: Optional[Path] = None
    if args.camera_settings == "":
        logging.info("Using camera settings from %s (--camera-settings '' passed)", config_path)
    else:
        camera_settings_path = resolve_recording_config_path(args.policy, args.camera_settings)
        if camera_settings_path is None:
            if args.camera_settings:
                logging.warning(
                    "Requested recording_config.yaml %s not found; falling back to %s",
                    args.camera_settings,
                    config_path,
                )
            else:
                logging.info(
                    "No recording_config.yaml found near %s; using %s",
                    args.policy,
                    config_path,
                )
        else:
            cfg = read_camera_overrides(cfg, camera_settings_path)

    if args.left:
        arm_keys_req = ["left"]
    elif args.right:
        arm_keys_req = ["right"]
    else:
        arm_keys_req = ["left", "right"]

    _controls = {c.strip() for c in args.recording_control.split(",")}
    _valid = {"keyboard", "pedal"}
    _unknown = _controls - _valid
    if _unknown:
        sys.exit(f"Unknown recording-control values: {_unknown}. Valid: {_valid}")

    use_keyboard = "keyboard" in _controls
    use_pedal = "pedal" in _controls
    use_reset_client = bool(cfg.get("use_reset_client", False))

    if use_pedal and not EVDEV_AVAILABLE:
        sys.exit("evdev is required for foot-pedal control. Install: pip install evdev")

    try:
        camera_configs = load_camera_configs(cfg, selected_arms=arm_keys_req)
        arm_configs = load_arm_configs(cfg)
    except ValueError as exc:
        sys.exit(str(exc))

    stabilize_flags = [c.stabilize_orientation_180 for c in camera_configs]
    orientation_thresholds = [c.orientation_mae_delta_threshold for c in camera_configs]
    if args.stabilize_orientation:
        stabilize_flags = [True] * len(camera_configs)
    elif args.no_stabilize_orientation:
        stabilize_flags = [False] * len(camera_configs)
    if args.orientation_mae_delta_threshold is not None:
        orientation_thresholds = [float(args.orientation_mae_delta_threshold)] * len(camera_configs)

    # ── Device / torch ──────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA not available, falling back to CPU")

    # ── Load policy (mirrors deploy.py) ─────────────────────────────────
    logging.info("Loading policy from %s...", args.policy)
    policy_type = _load_policy_type(args.policy, args.deploy_mode)
    if policy_type == "action_history_diffusion":
        if not _ACTION_HISTORY_AVAILABLE:
            raise RuntimeError(
                "Checkpoint declares type=action_history_diffusion but the "
                "lerobot_policy_action_history_diffusion plugin is not installed "
                "in this environment. Install it with "
                "`pip install -e packages/lerobot_policy_action_history_diffusion`."
            )
        policy = ActionHistoryDiffusionPolicy.from_pretrained(args.policy)
    elif policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(args.policy)
    else:
        policy = ACTPolicy.from_pretrained(args.policy)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    # Both DiffusionPolicy and its action_history subclass need their queues
    # primed via reset() before the first select_action() call.
    if policy_type in ("diffusion", "action_history_diffusion") and hasattr(policy, "reset"):
        policy.reset()

    strided_cfg = None
    strided_runner: Optional[StridedHistoryRunner] = None
    # action_history_diffusion has its own past-action queue layer inside the
    # policy, so it does NOT use StridedHistoryRunner -- the two extensions are
    # independent. Mirrors deploy.py.
    if policy_type == "diffusion":
        strided_cfg = load_strided_config(args.policy)
        if strided_cfg is not None:
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
            if int(round(args.hz)) != expected_hz:
                logging.warning(
                    "Strided history expects %d Hz inference; you requested --hz %.2f.",
                    expected_hz, args.hz,
                )
            logging.info(
                "Strided history enabled: n_obs_steps=%d, stride=%.2fs, buffer_length=%d frames",
                strided_runner.n_obs,
                strided_runner.stride_seconds,
                strided_runner.buffer_length,
            )
    logging.info("Policy loaded: %s on %s", policy.config.type, device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
    )

    image_feature_keys = _image_feature_keys(policy)
    expected_state_dim = policy.config.robot_state_feature.shape[0] if policy.config.robot_state_feature is not None else None
    policy_visual_size = _policy_visual_size(policy)

    training_resize = detect_training_resize(args.policy)
    resize_transform = None
    if training_resize:
        resize_transform = transforms_v2.Resize(list(training_resize), antialias=True)
        logging.info("Auto-detected training resize: %sx%s", training_resize[0], training_resize[1])
    elif policy_visual_size:
        resize_transform = transforms_v2.Resize(list(policy_visual_size), antialias=True)
        logging.info(
            "No training resize found; falling back to policy visual size: %sx%s",
            policy_visual_size[0], policy_visual_size[1],
        )

    per_camera_crops = detect_per_camera_crops(args.policy)
    if per_camera_crops:
        logging.info(
            "Auto-detected per-camera crops from train_config.json: %s",
            per_camera_crops,
        )

    fullres_crops = parse_fullres_crop_args(args.fullres_crop)
    if fullres_crops:
        logging.info("Using per-camera full-res crops from --fullres-crop: %s", fullres_crops)
    else:
        fullres_crops = detect_fullres_crops(args.policy)
        if fullres_crops:
            logging.info(
                "Auto-detected full-res crops from dataset info.json: %s",
                fullres_crops,
            )

    # ── Cameras ──────────────────────────────────────────────────────────
    logging.info("Opening configured cameras...")
    try:
        cameras = open_configured_cameras(camera_configs, override_resolution=args.resolution)
    except RuntimeError as exc:
        sys.exit(str(exc))
    time.sleep(0.3)
    logging.info("%d camera(s) active", len(cameras))

    # Map cameras to policy image keys BY NAME (not by position). ``cameras`` is
    # sorted alphabetically by name, which need not match the checkpoint's
    # ``input_features`` order; matching by name keeps each physical view on the
    # encoder slot it was trained with. See deploy.py for the full rationale.
    camera_index_by_feature_key: Dict[str, int] = {}
    for cam_idx, cam in enumerate(cameras):
        camera_index_by_feature_key[f"observation.images.{_slugify_camera_name(cam.name)}"] = cam_idx
    unmatched_policy_keys = [k for k in image_feature_keys if k not in camera_index_by_feature_key]
    if unmatched_policy_keys:
        logging.error(
            "No camera matches policy image key(s) %s by name. Available cameras: %s. "
            "Policy expects keys: %s.",
            unmatched_policy_keys,
            [cam.name for cam in cameras],
            image_feature_keys,
        )
        sys.exit(1)
    logging.info(
        "Mapping cameras to policy image keys by name: %s",
        {k: cameras[camera_index_by_feature_key[k]].name for k in image_feature_keys},
    )

    configs_by_name = {c.name: c for c in camera_configs}
    # Main-loop camera stabilizer is shared with recording so flipping stays
    # consistent across the policy/teleop boundary within an episode.
    main_stabilizer = CameraStabilizer(stabilize_flags, orientation_thresholds)

    # ── Dataset path / resume ───────────────────────────────────────────
    import shutil
    root = Path("data") / f"{args.name}_dagger"
    repo_id = f"ur_robotiq/{args.name}_dagger"
    n_threads = max(4 * len(cameras), 4)
    stats_path = root / "meta" / "recording_stats.json"

    _info_path = root / "meta" / "info.json"
    _can_resume = False
    if _info_path.exists():
        try:
            _info = json.loads(_info_path.read_text())
            _can_resume = _info.get("total_episodes", 0) > 0
        except (json.JSONDecodeError, KeyError):
            pass
    if not _can_resume and root.exists():
        shutil.rmtree(root)
        logging.info("Removed incomplete dataset at %s", root)

    recording_stats = RecordingStats.load(stats_path) if _can_resume else RecordingStats()

    # ── ROS 2 node + clients ────────────────────────────────────────────
    rclpy.init()
    node = rclpy.create_node("dagger_recorder")

    svc_cfg = cfg.get("services", {})
    control_mode_services = resolve_control_mode_services(svc_cfg)
    transition_ready_services = resolve_transition_ready_services(svc_cfg, control_mode_services)
    reset_service_names = resolve_reset_services(svc_cfg)
    control_mode_client = GelloControlModeClient(node, control_mode_services)
    transition_ready_client = GelloTransitionReadyClient(node, transition_ready_services)
    reset_service_client = ResetServiceClient(node, reset_service_names)
    reset_request_client = ResetRequestClient(node, ["/reset"]) if use_reset_client else None

    dagger_state = DaggerState()

    # Pick active arms from the experiment config + what the policy expects.
    experiment_config_path = resolve_experiment_config_path(args.policy, args.experiment_config)
    experiment_config = load_experiment_home_config(experiment_config_path) if experiment_config_path else None
    if experiment_config_path:
        logging.info("Using experiment config: %s", experiment_config_path)

    # Arm state listeners: for policy rollout we need state (observation),
    # and during teleop we need the action the operator is producing (which
    # gello_offset_node publishes to the same action_topic we publish to in
    # POLICY_ROLLOUT). ROSArmStateListener already subscribes to both.
    arm_states: Dict[str, ArmState] = {}
    listeners: Dict[str, ROSArmStateListener] = {}
    for key in arm_keys_req:
        arm_states[key] = ArmState()
        arm_cfg = arm_configs[key]
        listeners[key] = ROSArmStateListener(
            node,
            action_topic=arm_cfg.action_topic,
            state_topic=arm_cfg.state_topic,
            arm_state=arm_states[key],
            label=key,
        )

    try:
        wrap_joint_suffixes = resolve_wrap_joints(cfg)
    except ValueError as exc:
        sys.exit(f"Invalid wrap_joints in {config_path}: {exc}")
    unwrap_opts = resolve_unwrap_config(cfg)

    def _home_state_source(arm_name: str) -> Optional[np.ndarray]:
        arm_state = arm_states.get(arm_name)
        if arm_state is None:
            return None
        _, state_qpos, _ = arm_state.snapshot()
        if state_qpos is None:
            listener = listeners.get(arm_name)
            if listener is None:
                return None
            _, positions = listener.initial_state()
            return np.asarray(positions, dtype=np.float64) if positions else None
        return np.asarray(state_qpos, dtype=np.float64)

    home_sender = RobotHomeSender(
        node,
        arm_configs,
        state_source=_home_state_source,
        wrap_joint_suffixes=wrap_joint_suffixes,
        unwrap_max_step=unwrap_opts["max_step"],
        unwrap_waypoint_stamp_s=unwrap_opts["waypoint_stamp_s"],
        unwrap_settle_tolerance=unwrap_opts["settle_tolerance"],
        unwrap_settle_timeout_s=unwrap_opts["settle_timeout_s"],
    )

    # Spin ROS in background so the listeners populate ArmState.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    stop_event = threading.Event()

    _shutting_down = False
    _force_count = 0

    def _shutdown(sig: int, frame: object) -> None:
        nonlocal _shutting_down, _force_count
        if _shutting_down:
            _force_count += 1
            if _force_count >= 2:
                logging.warning("Force exit (data may be incomplete)")
                os._exit(1)
        _shutting_down = True
        logging.info("Shutting down (finalizing dataset)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logging.info("Waiting for action and state topics...")
    while not stop_event.is_set():
        if all(l.is_ready for l in listeners.values()):
            break
        time.sleep(0.1)

    if not stop_event.is_set():
        # Start in IDLE mode: we don't want GELLO to publish before policy takeover.
        control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label="init")

    if stop_event.is_set():
        logging.warning("Shutdown before initialization")
        node.destroy_node()
        rclpy.try_shutdown()
        return

    experiment_path = root / "experiment_config.yaml"
    disk_experiment_config = load_experiment_home_config(experiment_path)
    should_save_experiment_config = disk_experiment_config is None
    effective_experiment_config = disk_experiment_config or experiment_config

    def _listener_state(arm_name: str) -> Tuple[List[str], List[float]]:
        listener = listeners.get(arm_name)
        if listener is None:
            return [], []
        return listener.initial_state()

    home_positions, home_joint_names = resolve_home(
        effective_experiment_config, arm_keys_req, _listener_state
    )

    # Filter arm_keys down to those the policy actually expects (matches deploy.py).
    arm_keys = _resolve_arm_keys(arm_configs, home_joint_names, policy)
    if not arm_keys:
        sys.exit("Could not resolve active arms for policy.")
    if set(arm_keys) != set(arm_keys_req):
        logging.info("Policy expects arm(s) %s; using those (requested %s).", arm_keys, arm_keys_req)

    wrap_manager = WrapJointManager(wrap_joint_suffixes)
    wrap_manager.configure({arm: home_joint_names.get(arm, []) for arm in arm_keys})

    # Dataset create/resume
    features = build_features(arm_keys, home_joint_names, cameras, resolution=args.resolution)
    if _can_resume:
        logging.info("Resuming existing dataset at %s", root)
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=root,
            image_writer_threads=n_threads,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(args.hz),
            features=features,
            robot_type="ur3e_bimanual",
            root=root,
            use_videos=True,
            image_writer_threads=n_threads,
        )

    readers_by_name = {reader.name: reader for reader in cameras}
    resolved_cfg = build_recording_snapshot(cfg, camera_configs, readers_by_name)
    resolved_cfg["dagger"] = {
        "policy_path": str(args.policy),
        "prebuffer_seconds_override": args.prebuffer_seconds,
        "prebuffer_margin_frames": int(args.prebuffer_margin_frames),
    }
    with open(root / "recording_config.yaml", "w") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False)

    if should_save_experiment_config:
        save_experiment_home_config(
            experiment_path,
            home_positions,
            home_joint_names,
            wrap_joints=wrap_joint_suffixes,
        )

    dagger_state.episode_idx = dataset.meta.total_episodes

    # Policy action publishers (one per active arm). Used only while
    # ``dagger_state.policy_publish_enabled`` is set (i.e. during
    # POLICY_ROLLOUT). GELLO takes over the same topic during TELEOP.
    cmd_publishers: Dict[str, ROSActionPublisher] = {}
    for key in arm_keys:
        arm_cfg = arm_configs[key]
        cmd_publishers[key] = ROSActionPublisher(
            node,
            arm_cfg.action_topic,
            home_joint_names[key],
            label=key,
        )

    expected_action_dim = sum(len(home_joint_names[key]) for key in arm_keys)
    if policy_type in ("diffusion", "action_history_diffusion") and expected_state_dim is not None and expected_state_dim != expected_action_dim:
        logging.warning(
            "Policy expects a %s-D state/action, but the configured arms expose %s joints.",
            expected_state_dim, expected_action_dim,
        )

    # Pre-intervention ring buffer size: n_obs_steps × stride + margin, or
    # the CLI override if passed. Bigger is safer; the cost is just memory.
    prebuffer_frames = _resolve_prebuffer_frames(
        policy,
        strided_cfg,
        args.hz,
        args.prebuffer_seconds,
        args.prebuffer_margin_frames,
    )
    prebuffer = PreInterventionBuffer(max_frames=prebuffer_frames)
    logging.info(
        "Pre-intervention buffer: %d frames (≈ %.2fs at %.1f Hz)",
        prebuffer.max_frames,
        prebuffer.max_frames / max(args.hz, 1e-6),
        args.hz,
    )

    _banner = (
        f"\nDAgger recorder ready — {len(arm_keys)} arm(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        f"  Pre-intervention buffer: {prebuffer.max_frames} frames\n"
    )
    if use_keyboard:
        _banner += (
            "  Keyboard: 's' cycle phase (policy→freeze→teleop→save) | "
            "'d' discard/reset | 'r' reset | 'm' cycle mode | 'q' quit\n"
        )
    if use_pedal:
        _banner += (
            "  Pedal:    LEFT cycle phase (1:start policy, 2:freeze, 3:handover, 4:save) | "
            "MIDDLE discard/reset | RIGHT cycle mode\n"
        )
    _banner += (
        f"  GELLO mode: {GelloControlModeClient.mode_label(control_mode_client.mode)} "
        f"({control_mode_client.mode})\n"
    )
    logging.info(_banner)
    logging.info("Dataset -> %s  (episodes so far: %d)", root.resolve(), dataset.meta.total_episodes)
    logging.info(
        "State/action dim: %d, cameras: %d, hz: %s",
        len(features["action"]["names"]), len(cameras), args.hz,
    )

    # ── Command executor ────────────────────────────────────────────────
    command_executor = UserCommandExecutor()
    mode_cycle_lock = threading.Lock()
    mode_cycle_pending = 0
    mode_cycle_worker_running = False

    def _reset_policy_runtime() -> None:
        """Clear all per-episode policy state (matches deploy._start_new_episode)."""
        if strided_runner is not None:
            strided_runner.reset()
        elif hasattr(policy, "reset"):
            policy.reset()
        if hasattr(preprocessor, "reset"):
            preprocessor.reset()
        if hasattr(postprocessor, "reset"):
            postprocessor.reset()
        wrap_manager.reset_episode()
        main_stabilizer.reset()

    def _bounded_wait_for_resume(label: str, timeout_s: float = 5.0) -> bool:
        done = threading.Event()
        result_holder = [False]

        def _waiter() -> None:
            try:
                result_holder[0] = transition_ready_client.wait_for_resume(label=label)
            finally:
                done.set()

        threading.Thread(
            target=_waiter,
            name=f"transition-wait-{label}",
            daemon=True,
        ).start()
        if not done.wait(timeout=timeout_s):
            logging.warning(
                "[%s] transition-ready ack did not arrive within %.1fs; "
                "continuing handover anyway.",
                label,
                timeout_s,
            )
            return False
        return result_holder[0]

    def _submit_phase_toggle(source: str) -> None:
        """Pedal 1 / key 's': cycle IDLE → POLICY → FROZEN → TELEOP → (save) → IDLE."""
        def _action() -> None:
            phase = dagger_state.phase
            if phase == Phase.IDLE:
                _reset_policy_runtime()
                # Seed wrap-joint offsets from the home pose like deploy.py so
                # the first state snapshot can't race the ROS callback.
                ep_idx = dagger_state.episode_idx + 1
                for arm_name in arm_keys:
                    home_arr = home_positions.get(arm_name) or []
                    if home_arr:
                        wrap_manager.prelatch_offset(arm_name, home_arr, ep_idx)
                prebuffer.clear()
                dagger_state.begin_policy_rollout(label=source)
            elif phase == Phase.POLICY_ROLLOUT:
                # Freeze: silence the policy publisher and stop pushing to the
                # prebuffer. GELLO stays IDLE so the robot holds its last pose
                # while the operator gets set for handover. Frozen frames are
                # intentionally NOT recorded into the pre-intervention window.
                dagger_state.begin_frozen(label=source)
            elif phase == Phase.FROZEN:
                # Handover: switch GELLO to NORMAL, wait for transition-ready
                # so the first teleop frame we record actually comes from
                # GELLO. The main loop will flush the prebuffer (captured
                # during POLICY_ROLLOUT) as soon as phase == TELEOP.
                # Pause capture across the transition; run set_mode +
                # wait_for_resume in a background thread so the command
                # executor lock is not held (otherwise pedal-4 save presses
                # are dropped until the ack arrives).
                if not dagger_state.begin_teleop(label=source):
                    return
                dagger_state.set_capture_paused(True)
                logging.info("[%s] capture PAUSED for handover transition", source)

                def _handover_worker() -> None:
                    try:
                        control_mode_client.set_mode(
                            GelloControlModeClient.MODE_NORMAL, label=source,
                        )
                        _bounded_wait_for_resume(label=source, timeout_s=5.0)
                    finally:
                        dagger_state.set_capture_paused(False)
                        logging.info("[%s] capture RESUMED after handover transition", source)

                threading.Thread(
                    target=_handover_worker,
                    name=f"dagger-handover-{source}",
                    daemon=True,
                ).start()
            elif phase == Phase.TELEOP_CORRECTION:
                # End: flip phase to SAVING so the recording thread persists
                # what it has, then put GELLO back to IDLE.
                dagger_state.end_episode(label=source, save=True)
                control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=source)
            else:
                logging.info("[%s] Ignored toggle: phase=%s", source, phase.name)

        command_executor.submit(source, "phase_toggle", _action)

    def _submit_discard(source: str) -> None:
        def _action() -> None:
            dagger_state.discard(label=source)
            control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=source)
            dagger_state.set_transient_label("DISCARDED", duration_s=0.9)

        command_executor.submit(source, "discard", _action)

    def _open_gripper_before_home(arm_name: str, label: str) -> None:
        """Publish a JointState that keeps arm joints at their current state
        but moves the gripper joint to its home value.

        This is the first step of the pedal-2 reset sequence: it lets go of
        whatever the gripper is holding *before* the arm is told to move home.
        Without it, an arm closed around an object will fight the home
        trajectory and trigger a protective stop that needs a robot reboot.
        """
        cfg = arm_configs.get(arm_name)
        names = home_joint_names.get(arm_name, [])
        home = home_positions.get(arm_name, [])
        if cfg is None or not names or not home:
            return

        gripper_suffix = cfg.gripper_joint  # e.g. "robotiq_85_left_knuckle_joint"
        # Match by suffix so left/right prefixes (e.g. "left_robotiq_...") resolve.
        try:
            g_idx = next(i for i, n in enumerate(names) if n.endswith(gripper_suffix))
        except StopIteration:
            logging.warning(
                "[%s] gripper joint suffix %r not found in home_joint_names; "
                "skipping gripper-open pre-step",
                arm_name, gripper_suffix,
            )
            return

        current = _home_state_source(arm_name)
        if current is None or len(current) < len(names):
            logging.warning(
                "[%s] current state unavailable; skipping gripper-open pre-step",
                arm_name,
            )
            return

        positions = [float(current[i]) for i in range(len(names))]
        positions[g_idx] = float(home[g_idx])

        # Reuse the JointState publisher that home_sender already owns so we
        # don't race with it on the same action topic.
        pub = home_sender._publishers.get(arm_name)
        if pub is None:
            return

        msg = JointStateMsg()
        msg.header.stamp.sec = 3
        msg.header.stamp.nanosec = 0
        msg.name = list(names)
        msg.position = positions
        pub.publish(msg)
        logging.info(
            "[%s] [%s] Gripper-open published (idx %d, target %.4f); holding arm at current state",
            label, arm_name, g_idx, positions[g_idx],
        )
        time.sleep(3.0)

    # Tolerances for the skip-if-already-home / verify-reached-home logic
    # below. Values are in radians on non-gripper joints.
    #   * SKIP threshold: tight — only skip the home publish if the arm is
    #     essentially already there. Leaves room for normal motion commands.
    #   * VERIFY threshold: looser — we expect the arm to be within this much
    #     of home after the 10-s sleep the home publisher takes. If it isn't,
    #     the controller didn't act on the command (protective stop, inactive
    #     controller, RTDE shutdown, etc.).
    _HOME_SKIP_TOL_RAD = 0.02
    _HOME_VERIFY_TOL_RAD = 0.08

    def _arm_gripper_index(arm_name: str) -> int:
        """Return the index of the gripper joint in home_joint_names[arm_name],
        or -1 if none can be resolved."""
        cfg = arm_configs.get(arm_name)
        names = home_joint_names.get(arm_name, [])
        if cfg is None or not names:
            return -1
        suffix = cfg.gripper_joint
        for i, n in enumerate(names):
            if n.endswith(suffix):
                return i
        return -1

    def _arm_non_gripper_max_delta(arm_name: str) -> Optional[float]:
        """Max-abs error (rad) between current arm state and the home target,
        across all NON-gripper joints. ``None`` if state or config is missing.

        Used by the reset path to (a) skip home publishes for arms already at
        home and (b) verify after the home sleep that the arm actually moved.
        """
        names = home_joint_names.get(arm_name, [])
        home = home_positions.get(arm_name, [])
        if not names or not home or len(home) < len(names):
            return None
        current = _home_state_source(arm_name)
        if current is None or len(current) < len(names):
            return None
        g_idx = _arm_gripper_index(arm_name)
        max_delta = 0.0
        for i in range(len(names)):
            if i == g_idx:
                continue
            if i >= len(current):
                continue
            max_delta = max(max_delta, abs(float(current[i]) - float(home[i])))
        return max_delta

    def _submit_reset(source: str) -> None:
        def _action() -> None:
            if dagger_state.is_recording:
                logging.warning("[%s] Reset requested while recording; ignoring", source)
                return
            dagger_state.set_resetting(True)
            try:
                control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=source)
                if use_reset_client and reset_request_client is not None:
                    reset_request_client.call(label=source)
                else:
                    # Sequence:
                    #   1. open right gripper
                    #   2. open left gripper
                    #   3. home any arm that isn't already at home
                    #   4. verify each arm ended up at home
                    reset_arm_order = [a for a in ("right", "left") if a in home_positions]
                    for arm in reset_arm_order:
                        _open_gripper_before_home(arm, label=source)

                    # Skip the home publish for arms already essentially at
                    # home. Republishing a zero-delta trajectory is wasted
                    # wall-clock at best and, on the custom UR bridge we've
                    # seen misbehave, can put the controller in a state where
                    # the *next* non-zero command is ignored too.
                    homing_arms: List[str] = []
                    for arm in reset_arm_order:
                        delta = _arm_non_gripper_max_delta(arm)
                        if delta is None:
                            # State unavailable — be safe and publish.
                            homing_arms.append(arm)
                            continue
                        if delta < _HOME_SKIP_TOL_RAD:
                            logging.info(
                                "[%s] [%s] already at home (max_delta=%.4f rad); "
                                "skipping home publish",
                                source, arm, delta,
                            )
                            continue
                        homing_arms.append(arm)

                    if homing_arms:
                        ordered_home_positions = {a: home_positions[a] for a in homing_arms}
                        ordered_home_joint_names = {a: home_joint_names[a] for a in homing_arms}
                        home_sender.send_home(ordered_home_positions, ordered_home_joint_names)

                    reset_service_client.call(label=source)

                    # Verify every arm ended up at home. An arm whose home
                    # publish we skipped should still be at home; an arm whose
                    # home publish ran should be at home after send_home's
                    # per-arm 10-s sleep. If not, the controller didn't act —
                    # surface that immediately so the operator knows it's a
                    # robot-side issue and no amount of pedal pressing will fix
                    # it.
                    any_stuck = False
                    for arm in reset_arm_order:
                        delta = _arm_non_gripper_max_delta(arm)
                        if delta is None:
                            logging.warning(
                                "[%s] [%s] post-reset verification skipped — state unavailable",
                                source, arm,
                            )
                            continue
                        if delta > _HOME_VERIFY_TOL_RAD:
                            any_stuck = True
                            logging.error(
                                "[%s] [%s] DID NOT REACH HOME after reset "
                                "(max_delta=%.3f rad, tol=%.3f). The arm "
                                "controller on ur_robotiq likely did not "
                                "execute the trajectory — possible protective "
                                "stop, inactive controller, or dead hardware "
                                "interface. Check the ur_robotiq container "
                                "logs and `ros2 control list_hardware_components`. "
                                "Pedal presses will NOT recover this; a driver "
                                "restart is required.",
                                source, arm, delta, _HOME_VERIFY_TOL_RAD,
                            )
                        else:
                            logging.info(
                                "[%s] [%s] at home (max_delta=%.4f rad)",
                                source, arm, delta,
                            )
                    if any_stuck:
                        logging.error(
                            "[%s] Reset finished with at least one arm NOT at home. "
                            "Recording further episodes is unsafe until the arm "
                            "controller is recovered.",
                            source,
                        )
            finally:
                dagger_state.set_resetting(False)
            logging.info("[%s] Reset complete; ready for next episode", source)

        command_executor.submit(source, "reset", _action)

    def _submit_discard_or_reset(source: str) -> None:
        # Matches record.py's middle-pedal: discard while recording, otherwise home/reset.
        if dagger_state.is_recording:
            _submit_discard(source)
        else:
            _submit_reset(source)

    def _submit_cycle_mode(source: str) -> None:
        nonlocal mode_cycle_pending, mode_cycle_worker_running

        if dagger_state.is_recording:
            dagger_state.set_capture_paused(True)
            logging.info("[%s] capture PAUSED for mode cycle", source)

        with mode_cycle_lock:
            mode_cycle_pending += 1
            should_start_worker = not mode_cycle_worker_running
            if should_start_worker:
                mode_cycle_worker_running = True

        if should_start_worker:
            threading.Thread(
                target=_cycle_mode_with_pause,
                args=(source,),
                name=f"mode-cycle-{source}",
                daemon=True,
            ).start()

    def _cycle_mode_with_pause(source: str) -> None:
        nonlocal mode_cycle_pending, mode_cycle_worker_running

        transition_confirmed = True
        try:
            while True:
                with mode_cycle_lock:
                    if mode_cycle_pending <= 0:
                        break
                    mode_cycle_pending -= 1

                next_mode = GelloControlModeClient.cycle_mode(control_mode_client.mode)
                control_mode_client.set_mode(next_mode, label=source)
                transition_confirmed = transition_ready_client.wait_for_resume(label=source)
                if not transition_confirmed:
                    logging.warning(
                        "[%s] Transition completion not confirmed; keeping capture paused",
                        source,
                    )
                    break
        finally:
            with mode_cycle_lock:
                mode_cycle_worker_running = False
                queued_left = mode_cycle_pending

            if transition_confirmed and queued_left == 0:
                dagger_state.set_capture_paused(False)
                logging.info("[%s] capture RESUMED after mode cycle", source)

    # Optional service so external scripts can force a reset (matches record.py).
    reset_return_service = None
    if use_reset_client:

        def _handle_reset_return(request: object, response: object):
            _submit_reset("reset_return")
            response.success = True
            response.message = "reset scheduled"
            return response

        reset_return_service = node.create_service(TriggerSrv, "/reset_return", _handle_reset_return)
        logging.info("Reset return service available at /reset_return")

    # ── Unified control/recording loop ──────────────────────────────────
    period = 1.0 / float(args.hz)
    step_count = 0
    teleop_frames_in_episode = 0
    record_h, record_w = args.resolution if args.resolution else (None, None)

    def _build_image_features_from_frames(
        frames: List[Optional[np.ndarray]],
    ) -> Tuple[Dict[str, "Image.Image"], Dict[str, torch.Tensor], bool]:
        """Return (dataset-side PIL images, policy-side tensors, all_ok).

        Dataset gets the original-resolution BGR→RGB PIL frames (so saved
        videos match the camera's native size), while the policy gets
        resized float tensors. Missing frames → all_ok == False.
        """
        dataset_images: Dict[str, Image.Image] = {}
        policy_images: Dict[str, torch.Tensor] = {}
        all_ok = True
        for i, cam in enumerate(cameras):
            bgr = frames[i]
            feature_key = cam.feature_key
            if bgr is None:
                all_ok = False
                continue
            dataset_bgr = bgr
            if record_h is not None:
                dataset_bgr = cv2.resize(dataset_bgr, (record_w, record_h))
            dataset_images[feature_key] = Image.fromarray(cv2.cvtColor(dataset_bgr, cv2.COLOR_BGR2RGB))

            policy_key = f"observation.images.{_slugify_camera_name(cam.name)}"
            if policy_key in image_feature_keys:
                policy_bgr = bgr
                if fullres_crops:
                    policy_bgr = apply_fullres_crop(policy_bgr, policy_key, fullres_crops)
                if resize_transform is None and policy_visual_size is not None:
                    policy_bgr = cv2.resize(
                        policy_bgr,
                        (policy_visual_size[1], policy_visual_size[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(policy_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                if resize_transform is not None:
                    tensor = resize_transform(tensor)
                if per_camera_crops:
                    tensor = apply_per_camera_crop(
                        tensor, policy_key, per_camera_crops
                    )
                policy_images[policy_key] = tensor
        return dataset_images, policy_images, all_ok

    def _flush_prebuffer() -> int:
        """Write the pre-intervention ring buffer into the dataset.

        Called on the first TELEOP tick right before we start appending
        human frames. Empty buffer is a no-op.
        """
        flushed = 0
        for frame in prebuffer.drain():
            try:
                dataset.add_frame(frame)
                flushed += 1
            except Exception:
                logging.exception("Failed to flush pre-buffer frame; aborting episode")
                dataset.clear_episode_buffer()
                dagger_state.discard(label="recorder")
                return 0
        if flushed:
            logging.info("Pre-buffer flushed: %d frames (action_source=0)", flushed)
        return flushed

    def _coerce_scalar_episode_columns() -> None:
        """Work around a LeRobot + numpy-2.x mismatch on shape-(1,) features.

        LeRobot declares our ``action_source`` feature with ``shape=(1,)``.
        ``get_hf_features_from_features`` special-cases that shape as a scalar
        ``datasets.Value(dtype="int64")`` column (feature_utils.py), but
        ``dataset.add_frame`` validates each entry as an ndarray of shape (1,).
        At ``save_episode`` time ``Dataset.from_dict`` calls ``int(value)`` for
        each row; numpy 2.x no longer allows that on a 1-D ndarray and raises
        ``TypeError: only 0-dimensional arrays can be converted to Python
        scalars``. Flatten the buffered values to Python ints right before the
        save so the HF ``Value`` encoder is happy.
        """
        buf = getattr(dataset, "writer", None)
        if buf is None:
            return
        ep_buffer = getattr(buf, "episode_buffer", None)
        if not ep_buffer:
            return
        if "action_source" in ep_buffer:
            ep_buffer["action_source"] = [
                int(np.asarray(v).reshape(-1)[0]) for v in ep_buffer["action_source"]
            ]

    def _drain_save_request() -> None:
        nonlocal teleop_frames_in_episode
        save_flag, discard_flag = dagger_state.take_save_request()
        if not save_flag:
            return
        if discard_flag:
            dataset.clear_episode_buffer()
            recording_stats.record_discard()
            recording_stats.write(stats_path, args.hz)
            logging.info("Episode discarded")
            prebuffer.clear()
            teleop_frames_in_episode = 0
            return
        if teleop_frames_in_episode <= 0:
            logging.warning("No frames in episode; nothing to save")
            dataset.clear_episode_buffer()
            recording_stats.record_discard()
            recording_stats.write(stats_path, args.hz)
            prebuffer.clear()
            return
        _coerce_scalar_episode_columns()
        try:
            dataset.save_episode()
            recording_stats.record_save()
            logging.info(
                "Episode %d saved (%d human frames)",
                dagger_state.episode_idx - 1,
                teleop_frames_in_episode,
            )
        except Exception:
            logging.exception(
                "Failed to save episode %d; dropping buffered data",
                dagger_state.episode_idx - 1,
            )
            dataset.clear_episode_buffer()
            recording_stats.record_discard()
        finally:
            recording_stats.write(stats_path, args.hz)
            prebuffer.clear()
            teleop_frames_in_episode = 0

    # ── Visualization / overlay ─────────────────────────────────────────
    preview: Optional[LivePreview] = None
    cam_labels: List[str] = []
    if cameras and args.visualize:
        cam_labels = [c.display_label if getattr(c, "display_label", "") else f"cam {c.index}" for c in cameras]

    def _draw_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
        phase = dagger_state.phase
        transient = dagger_state.get_transient_label()
        if transient is not None:
            label = f"{transient}  ep {dagger_state.episode_idx}"
            color = (80, 220, 255)
        elif phase == Phase.IDLE:
            if dagger_state.is_resetting:
                label = f"RESET  ep {dagger_state.episode_idx}"
                color = (0, 200, 255)
            else:
                label = f"IDLE  ep {dagger_state.episode_idx}  (pedal 1 → start policy)"
                color = (180, 180, 180)
        elif phase == Phase.POLICY_ROLLOUT:
            label = (
                f"POLICY  ep {dagger_state.episode_idx}  "
                f"buf {len(prebuffer)}/{prebuffer.max_frames}  step {step_count}"
            )
            color = (0, 220, 0)
        elif phase == Phase.FROZEN:
            label = (
                f"FROZEN  ep {dagger_state.episode_idx}  "
                f"buf {len(prebuffer)}/{prebuffer.max_frames}  (pedal 1 → handover)"
            )
            color = (0, 200, 255)
        elif phase == Phase.TELEOP_CORRECTION:
            label = (
                f"CORRECTING  ep {dagger_state.episode_idx}  "
                f"human-frames {teleop_frames_in_episode}"
            )
            color = (80, 80, 255)
        else:  # SAVING
            label = f"SAVING  ep {dagger_state.episode_idx}"
            color = (0, 180, 220)

        cv2.putText(canvas, label, (20, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(canvas, label, (20, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        mode_text = (
            f"MODE {GelloControlModeClient.mode_label(control_mode_client.mode)} "
            f"({control_mode_client.mode})"
        )
        if control_mode_client.mode == GelloControlModeClient.MODE_IDLE:
            dot_color = (0, 0, 255)
        else:
            dot_color = (0, 200, 0)
        mode_origin = (disp_w - 260, disp_h - 16)
        dot_center = (mode_origin[0] + 12, mode_origin[1] - 12)
        mode_text_origin = (mode_origin[0] + 30, mode_origin[1])
        cv2.circle(canvas, dot_center, 8, (0, 0, 0), 3)
        cv2.circle(canvas, dot_center, 6, dot_color, -1)
        cv2.putText(canvas, mode_text, mode_text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(canvas, mode_text, mode_text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 230, 255), 1)

    # ── Pedal thread ────────────────────────────────────────────────────
    if use_pedal:
        pedal_thread = FootPedalThread(
            device_path=args.pedal_device,
            stop_event=stop_event,
            on_toggle=lambda: _submit_phase_toggle("pedal"),
            on_discard=lambda: _submit_discard_or_reset("pedal"),
            on_reset=lambda: _submit_cycle_mode("pedal"),
        )
        pedal_thread.start()

    # ── Terminal keyboard (only when --visualize is off; otherwise the
    # preview window forwards keys).
    terminal_keyboard_enabled = use_keyboard and not args.visualize
    if terminal_keyboard_enabled:
        threading.Thread(
            target=run_cbreak_keyboard_loop,
            args=(
                stop_event,
                {
                    "s": lambda: _submit_phase_toggle("keyboard"),
                    "d": lambda: _submit_discard_or_reset("keyboard"),
                    "r": lambda: _submit_reset("keyboard"),
                    "m": lambda: _submit_cycle_mode("keyboard"),
                },
            ),
            name="kb-loop",
            daemon=True,
        ).start()

    # ── Main loop ───────────────────────────────────────────────────────
    try:
        with contextlib.ExitStack() as stack:
            if cameras and args.visualize:
                preview = stack.enter_context(
                    LivePreview(
                        window_name=(
                            "dagger.py — S phase | D discard/reset | R reset | M cycle mode | Q quit"
                            if use_keyboard else
                            "dagger.py — live preview (record via pedal) | Q quit"
                        ),
                        cam_labels=cam_labels,
                        fullscreen=False,
                        wait_key_ms=1,
                    )
                )

            while not stop_event.is_set():
                t_start = time.monotonic()

                frames = main_stabilizer.process([cam.get_frame() for cam in cameras])

                # 1. Always read arm state for observation + dataset.
                state_parts_raw: List[Optional[np.ndarray]] = []
                action_parts_gello: List[Optional[np.ndarray]] = []
                all_state_ready = True
                for key in arm_keys:
                    arm_state = arm_states.get(key)
                    if arm_state is None:
                        all_state_ready = False
                        state_parts_raw.append(None)
                        action_parts_gello.append(None)
                        continue
                    action_qpos, state_qpos, _ = arm_state.snapshot()
                    state_parts_raw.append(state_qpos)
                    action_parts_gello.append(action_qpos)
                    if state_qpos is None:
                        all_state_ready = False

                phase = dagger_state.phase
                _drain_save_request()  # harmless when no save pending

                # Idle / frozen / saving path: just render + sleep.
                # FROZEN explicitly skips policy inference and prebuffer pushes
                # so frames between pedal-1-press-2 (freeze) and pedal-1-press-3
                # (handover) never enter the saved episode.
                if phase == Phase.IDLE or phase == Phase.FROZEN or phase == Phase.SAVING:
                    if preview is not None:
                        key = preview.render(frames, _draw_overlay)
                        if key in (ord('q'), ord('Q'), 27):
                            stop_event.set()
                            break
                        if use_keyboard:
                            if key in (ord('s'), ord('S')):
                                _submit_phase_toggle("keyboard")
                            elif key in (ord('d'), ord('D')):
                                _submit_discard_or_reset("keyboard")
                            elif key in (ord('r'), ord('R')):
                                _submit_reset("keyboard")
                            elif key in (ord('m'), ord('M')):
                                _submit_cycle_mode("keyboard")
                    else:
                        time.sleep(0.05)
                    continue

                if not all_state_ready:
                    time.sleep(0.01)
                    continue

                # 2. Build observation (wrap-adjusted state + images).
                state_parts = [
                    wrap_manager.subtract_state(key, qpos_raw, max(dagger_state.episode_idx, 0) + 1)
                    for key, qpos_raw in zip(arm_keys, state_parts_raw)
                ]
                state_arr = np.concatenate(state_parts).astype(np.float32)
                if expected_state_dim is not None and state_arr.shape[0] != expected_state_dim:
                    logging.error(
                        "Expected a %s-D state vector, received %s-D. Check arm config.",
                        expected_state_dim, state_arr.shape[0],
                    )
                    time.sleep(0.05)
                    continue

                dataset_images, policy_images, images_ok = _build_image_features_from_frames(frames)
                if not images_ok:
                    time.sleep(0.01)
                    continue

                # 3. Branch on phase.
                if phase == Phase.POLICY_ROLLOUT:
                    observation: dict = {
                        "observation.state": torch.from_numpy(state_arr),
                    }
                    for k, v in policy_images.items():
                        observation[k] = v
                    missing = [k for k in image_feature_keys if k not in observation]
                    if missing:
                        logging.error("Missing policy observation images: %s", missing)
                        time.sleep(0.05)
                        continue
                    if policy_type in ("diffusion", "action_history_diffusion"):
                        observation = {k: observation[k] for k in (image_feature_keys + ["observation.state"])}

                    processed_obs = preprocessor(observation)
                    with torch.inference_mode():
                        if strided_runner is not None:
                            action = strided_runner.select_action(processed_obs)
                        else:
                            action = policy.select_action(processed_obs)
                    action = postprocessor(action)
                    action_np = action.squeeze(0).cpu().numpy()

                    if action_np.size < expected_action_dim:
                        logging.error(
                            "Policy action dim %s smaller than expected %s",
                            action_np.size, expected_action_dim,
                        )
                        time.sleep(0.05)
                        continue

                    # Publish to the arm(s). We save the *policy's raw*
                    # action (training coordinates, pre-wrap) as the
                    # dataset action so the pre-buffer row is consistent
                    # with what record.py writes for teleop frames (which
                    # are also wrap-subtracted). ``add_action`` only adds
                    # the wrap offset for the published-to-controller path.
                    slice_start = 0
                    for key in arm_keys:
                        joint_count = len(home_joint_names[key])
                        arm_action = action_np[slice_start:slice_start + joint_count].astype(np.float32).copy()
                        published_action = wrap_manager.add_action(key, arm_action)
                        if dagger_state.policy_publish_enabled.is_set():
                            cmd_publishers[key].send(published_action)
                        slice_start += joint_count
                    policy_action_for_dataset = action_np[:expected_action_dim].astype(np.float32)

                    # Build pre-buffer frame. Uses the same schema as
                    # dataset.add_frame for a seamless flush at handover.
                    frame_dict = {
                        "task": args.task,
                        "action": policy_action_for_dataset,
                        "observation.state": state_arr,
                        "action_source": np.asarray([ACTION_SOURCE_POLICY], dtype=np.int64),
                    }
                    for k, v in dataset_images.items():
                        frame_dict[k] = v
                    if not dagger_state.is_capture_paused:
                        prebuffer.push(frame_dict)

                    step_count += 1
                    if step_count % 100 == 0:
                        logging.info(
                            "Policy step %d — buffer %d/%d",
                            step_count, len(prebuffer), prebuffer.max_frames,
                        )

                elif phase == Phase.TELEOP_CORRECTION:
                    # First TELEOP tick: flush pre-buffer so the history window
                    # is aligned with the handover boundary.
                    if teleop_frames_in_episode == 0 and len(prebuffer) > 0:
                        _flush_prebuffer()

                    if any(a is None for a in action_parts_gello):
                        # GELLO hasn't caught up yet; skip this tick.
                        time.sleep(0.01)
                        continue

                    action_parts: List[np.ndarray] = []
                    state_parts_mod: List[np.ndarray] = []
                    for key, state_raw, action_raw in zip(arm_keys, state_parts_raw, action_parts_gello):
                        state_mod, action_mod = wrap_manager.subtract_state_action(
                            key, state_raw, action_raw, max(dagger_state.episode_idx, 0) + 1
                        )
                        state_parts_mod.append(state_mod.astype(np.float32))
                        action_parts.append(action_mod.astype(np.float32))

                    frame_dict = {
                        "task": args.task,
                        "action": np.concatenate(action_parts).astype(np.float32),
                        "observation.state": np.concatenate(state_parts_mod).astype(np.float32),
                        "action_source": np.asarray([ACTION_SOURCE_HUMAN], dtype=np.int64),
                    }
                    for k, v in dataset_images.items():
                        frame_dict[k] = v
                    if not dagger_state.is_capture_paused:
                        try:
                            dataset.add_frame(frame_dict)
                            teleop_frames_in_episode += 1
                        except Exception:
                            logging.exception("Failed to add teleop frame; dropping episode")
                            dataset.clear_episode_buffer()
                            dagger_state.discard(label="recorder")
                            teleop_frames_in_episode = 0

                # 4. Rendering.
                if preview is not None:
                    key = preview.render(frames, _draw_overlay)
                    if key in (ord('q'), ord('Q'), 27):
                        stop_event.set()
                        break
                    if use_keyboard:
                        if key in (ord('s'), ord('S')):
                            _submit_phase_toggle("keyboard")
                        elif key in (ord('d'), ord('D')):
                            _submit_discard_or_reset("keyboard")
                        elif key in (ord('r'), ord('R')):
                            _submit_reset("keyboard")
                        elif key in (ord('m'), ord('M')):
                            _submit_cycle_mode("keyboard")

                elapsed = time.monotonic() - t_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except Exception as e:
        logging.error("Main loop error: %s", e, exc_info=True)
        stop_event.set()
    finally:
        try:
            # If we're quitting mid-teleop, best-effort save what we have.
            if dagger_state.is_teleop and teleop_frames_in_episode > 0:
                logging.info("Saving in-progress episode before exit...")
                _coerce_scalar_episode_columns()
                dataset.save_episode()
                recording_stats.record_save()
                recording_stats.write(stats_path, args.hz)
                dagger_state.force_idle()
        except Exception as e:
            logging.warning("Could not save in-progress episode: %s", e)

        try:
            dataset.finalize()
            recording_stats.write(stats_path, args.hz)
            logging.info("Done — %d episode(s) -> %s", dagger_state.episode_idx, root.resolve())
        except Exception as e:
            logging.warning("Error finalizing dataset: %s", e)

        for cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass

        # Best-effort: flip GELLO back to IDLE so the next lerobot-ros run
        # inherits a sane state. Does not gate shutdown if the service is gone.
        try:
            control_mode_client.set_mode(
                GelloControlModeClient.MODE_IDLE, label="shutdown"
            )
        except Exception as e:
            logging.warning("Could not reset GELLO to IDLE on shutdown: %s", e)

        node.destroy_node()
        rclpy.try_shutdown()

    # ── Mirror the finished dagger dataset to Hugging Face (best-effort) ──
    # Never raises; enable with --push or LEROBOT_HF_PUSH=1.
    try_sync_to_hub(root, push=args.push)


if __name__ == "__main__":
    main()
