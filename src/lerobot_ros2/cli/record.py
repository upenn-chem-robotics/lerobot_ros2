#!/usr/bin/env python3
"""
Bimanual recording script for UR3e + GELLO — LeRobot dataset format.

Pure ROS 2 subscriber: listens to the joint command topics (actions already
computed by gello_offset_node) and UR3e state topics (observations), plus
USB cameras, and writes everything to a LeRobot v3 dataset ready for ACT or
Diffusion Policy training.

This script does NOT control any hardware. Teleop is handled by ur_robotiq's
gello_offset_node. This script only records.

Controls (keyboard):
    s — start / stop episode
    d — discard / reset (discard while recording, reset while idle)
    m — cycle GELLO mode (NORMAL -> ROTATE_CW -> ROTATE_CCW)
        In ROTATE_CW/ROTATE_CCW only the `rotation_arm` from the config rotates
        wrist_3; every other arm is frozen (holds all joints) until NORMAL.
    q — quit and finalize dataset
    r — reset while idle

Controls (foot pedal — PCsensor 3-pedal FootSwitch):
  Left   pedal (KEY_A) — start / stop episode
    Middle pedal (KEY_B) — discard / reset (discard while recording, reset while idle)
    Right  pedal (KEY_C) — cycle GELLO mode (1 -> 2 -> 3 -> 1)
  (The recorder grabs every matching evdev node — composite devices often expose
   several, e.g. event-if00 and event-if01; grabbing only one lets the other
   still type into the terminal.)

Usage:
    # Terminal 1: launch ur_robotiq with GELLO teleop
    ros2 launch ur_robotiq control_bimanual_ur3_robotiq.launch.py use_gello:=true ...

    # Terminal 2: run this recorder
    lerobot-ros-record --name pick_place --task "pick up the block"
    lerobot-ros-record --name pick_place --task "..." --visualize
    lerobot-ros-record ... --recording-control pedal --visualize
    lerobot-ros-record --name pick_place --task "..." --hz 15
    lerobot-ros-record --name pick_place --task "..." --left
    lerobot-ros-record --name pick_place --task "..." --recording-control pedal
    lerobot-ros-record --name pick_place --task "..." --recording-control keyboard,pedal
"""

import argparse
import math
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import yaml

os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)

from lerobot_ros2.config_paths import resolve_config_path
from lerobot_ros2.hub_sync import try_sync_to_hub
from lerobot_ros2.helper import (
    ArmState,
    CameraReader,
    CameraStabilizer,
    ManualFlip180,
    RobotHomeSender,
    ROSArmStateListener,
    WrapJointManager,
    build_recording_snapshot,
    load_camera_configs,
    load_experiment_home_config,
    load_arm_configs,
    load_config,
    open_configured_cameras,
    resolve_home,
    resolve_unwrap_config,
    resolve_wrap_joints,
    run_cbreak_keyboard_loop,
    save_experiment_home_config,
)
from lerobot_ros2.visualizer import LivePreview
from lerobot_ros2.helper import (
    EVDEV_AVAILABLE,
    FootPedalThread,
    PEDAL_DEFAULT_DEVICE,
    resolve_pedal_evdev_paths,
)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters as SetParametersSrv
    from sensor_msgs.msg import JointState as JointStateMsg
    from std_srvs.srv import Empty as EmptySrv
    from std_srvs.srv import Trigger as TriggerSrv
    try:
        # rclpy >= Iron exposes SignalHandlerOptions; older releases don't.
        # When unavailable we fall back to bare rclpy.init() and rely on our
        # own SIGINT handler winning the race (best effort).
        from rclpy.signals import SignalHandlerOptions
        _RCLPY_SIGNAL_HANDLER_OPTIONS_AVAILABLE = True
    except ImportError:
        SignalHandlerOptions = None  # type: ignore[assignment]
        _RCLPY_SIGNAL_HANDLER_OPTIONS_AVAILABLE = False
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    _RCLPY_SIGNAL_HANDLER_OPTIONS_AVAILABLE = False

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


def load_camera_grid_columns(cfg: dict) -> int:
    visualization_cfg = cfg.get("visualization") or {}
    return int(visualization_cfg.get("camera_grid_columns", 1))


# Minimum gap between two accepted subtask steps. Sits on top of
# FootPedalThread's own per-action debounce.
_SUBTASK_DEBOUNCE_S = 0.5


class RecordingState:
    """Controls episode recording lifecycle, safe to call from multiple threads."""

    def __init__(self, subtask_count: int = 0) -> None:
        self._lock = threading.Lock()
        self.is_recording = False
        self.episode_idx = 0
        self._save_requested = False
        self._last_toggle_time = 0.0
        # 0 disables subtask labelling entirely: no counter, no dataset column.
        self._subtask_count = max(int(subtask_count), 0)
        self._subtask_index = 0
        self._last_subtask_time = 0.0
        self._is_resetting = False
        self._capture_paused = False
        self._transition_pending = False
        self._transient_label: Optional[str] = None
        self._transient_until = 0.0
        self._subtask_flash_label: Optional[str] = None
        self._subtask_flash_until = 0.0
        self._countdown_label: Optional[str] = None
        self._countdown_until = 0.0

    def toggle(self, label: str = "") -> Optional[str]:
        with self._lock:
            if self.is_recording:
                self._stop_locked(label=label)
                return "stopped"
            self._start_locked(label=label)
            return "started"

    def start(self, label: str = "") -> bool:
        with self._lock:
            return self._start_locked(label=label)

    def stop(self, label: str = "") -> None:
        with self._lock:
            self._stop_locked(label=label)

    def discard(self, label: str = "") -> None:
        with self._lock:
            self._stop_locked(label=label, save_episode=False)

    def _start_locked(self, label: str = "") -> bool:
        now = time.monotonic()
        if now - self._last_toggle_time < 1.0:
            return False
        self._last_toggle_time = now
        self.is_recording = True
        # Every episode begins in subtask 1, so the operator's first pedal press
        # is the 1 -> 2 boundary rather than the entry into subtask 1.
        self._subtask_index = 0
        self._last_subtask_time = 0.0
        logging.info(f"[{label}] ● Recording STARTED — episode {self.episode_idx}")
        if self._subtask_count > 0:
            logging.info(f"[{label}] Subtask 1/{self._subtask_count}")
        return True

    def _stop_locked(self, label: str = "", save_episode: bool = True) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        self._save_requested = save_episode
        if save_episode:
            logging.info(f"[{label}] ■ Recording STOPPED — saving episode {self.episode_idx}")
        else:
            logging.info(f"[{label}] X Recording DISCARDED — dropping episode {self.episode_idx}")

    def take_save_request(self) -> bool:
        with self._lock:
            if self._save_requested:
                self._save_requested = False
                self.episode_idx += 1
                return True
            return False

    @property
    def subtask_count(self) -> int:
        with self._lock:
            return self._subtask_count

    @property
    def subtask_index(self) -> int:
        with self._lock:
            return self._subtask_index

    def subtask_snapshot(self) -> Tuple[int, int]:
        """``(index, count)`` read under one lock; count 0 means labelling is off."""
        with self._lock:
            return self._subtask_index, self._subtask_count

    def advance_subtask(self, label: str = "") -> Optional[int]:
        """Move to the next subtask; returns the new index, or None if unchanged."""
        return self._step_subtask(1, label=label)

    def retreat_subtask(self, label: str = "") -> Optional[int]:
        """Move back a subtask (to undo a stray press); None if unchanged."""
        return self._step_subtask(-1, label=label)

    def _step_subtask(self, delta: int, label: str = "") -> Optional[int]:
        with self._lock:
            if self._subtask_count <= 0:
                return None
            now = time.monotonic()
            # FootPedalThread already debounces per action, but composite pedals
            # echo the same press across several evdev nodes and a double-count
            # here mislabels a whole subtask's worth of frames.
            if now - self._last_subtask_time < _SUBTASK_DEBOUNCE_S:
                logging.info(f"[{label}] Subtask step ignored (debounced)")
                return None
            self._last_subtask_time = now
            target = self._subtask_index + delta
            if not 0 <= target < self._subtask_count:
                edge = "first" if delta < 0 else "last"
                logging.warning(
                    f"[{label}] Subtask step ignored — already on the {edge} subtask "
                    f"({self._subtask_index + 1}/{self._subtask_count})"
                )
                return None
            self._subtask_index = target
            return target

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

    @property
    def is_transition_pending(self) -> bool:
        with self._lock:
            return self._transition_pending

    def set_transition_pending(self, value: bool) -> None:
        with self._lock:
            self._transition_pending = value

    def set_transient_label(self, label: str, duration_s: float = 0.8) -> None:
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

    def clear_transient_label(self) -> None:
        with self._lock:
            self._transient_label = None
            self._transient_until = 0.0

    def set_subtask_flash(self, label: str, duration_s: float = 1.5) -> None:
        """Announce a subtask change in the preview, big and centered.

        Deliberately separate from the transient label: that one is drawn by a
        branch that sits *below* the ``is_recording`` branch, so it is
        unreachable mid-episode — exactly when a subtask change happens.
        """
        with self._lock:
            self._subtask_flash_label = label
            self._subtask_flash_until = time.monotonic() + max(duration_s, 0.0)

    def get_subtask_flash(self) -> Optional[str]:
        with self._lock:
            if self._subtask_flash_label is None:
                return None
            if time.monotonic() > self._subtask_flash_until:
                self._subtask_flash_label = None
                self._subtask_flash_until = 0.0
                return None
            return self._subtask_flash_label

    def set_countdown(self, label: str, duration_s: float = 5.0) -> None:
        with self._lock:
            self._countdown_label = label
            self._countdown_until = time.monotonic() + max(duration_s, 0.0)

    def get_countdown(self) -> tuple[Optional[str], float]:
        with self._lock:
            if self._countdown_label is None:
                return None, 0.0
            remaining = self._countdown_until - time.monotonic()
            if remaining <= 0.0:
                self._countdown_label = None
                self._countdown_until = 0.0
                return None, 0.0
            return self._countdown_label, remaining

    def clear_countdown(self) -> None:
        with self._lock:
            self._countdown_label = None
            self._countdown_until = 0.0


class UserCommandExecutor:
    """Runs at most one keyboard/pedal command at a time; drops overlapping inputs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def submit(self, source: str, action: str, fn: Callable[[], None]) -> bool:
        if not self._lock.acquire(blocking=False):
            logging.info("[%s] Ignored '%s' command: another command is in progress", source, action)
            return False

        def _runner() -> None:
            try:
                fn()
            except Exception:
                logging.exception("[%s] Command '%s' failed", source, action)
            finally:
                self._lock.release()

        threading.Thread(target=_runner, name=f"cmd-{source}-{action}", daemon=True).start()
        return True


class RecordingStats:
    """Tracks recording outcomes and persists them alongside the dataset."""

    def __init__(self, discarded_attempts: int = 0, saved_episodes: int = 0) -> None:
        self._lock = threading.Lock()
        self.discarded_attempts = int(discarded_attempts)
        self.saved_episodes = int(saved_episodes)

    def record_discard(self) -> None:
        with self._lock:
            self.discarded_attempts += 1

    def record_save(self) -> None:
        with self._lock:
            self.saved_episodes += 1

    def to_dict(self, recording_hz: float) -> dict:
        with self._lock:
            attempts = self.saved_episodes + self.discarded_attempts
            success_rate = self.saved_episodes / attempts if attempts else 0.0
            return {
                "saved_episodes": self.saved_episodes,
                "discarded_attempts": self.discarded_attempts,
                "total_attempts": attempts,
                "recording_hz": recording_hz,
                "success_rate": success_rate,
            }

    def write(self, path: Path, recording_hz: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(recording_hz)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "RecordingStats":
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            return cls()
        except json.JSONDecodeError:
            logging.warning("Could not parse recording stats at %s; starting fresh", path)
            return cls()

        return cls(
            discarded_attempts=int(payload.get("discarded_attempts", 0)),
            saved_episodes=int(payload.get("saved_episodes", 0)),
        )


# ── GELLO control mode client ─────────────────────────────────────────────


class GelloControlModeClient:
    """Sets GELLO control_mode via rcl_interfaces/srv/SetParameters."""

    MODE_IDLE = 0
    MODE_NORMAL = 1
    MODE_CLOCKWISE = 2
    MODE_COUNTERCLOCKWISE = 3
    _VALID_MODES = {MODE_IDLE, MODE_NORMAL, MODE_CLOCKWISE, MODE_COUNTERCLOCKWISE}

    def __init__(
        self,
        node: "Node",
        service_names: List[str],
        default_mode: int = MODE_NORMAL,
        rotation_services: Optional[List[str]] = None,
    ) -> None:
        self._node = node
        self._clients = {
            name: node.create_client(SetParametersSrv, name)
            for name in service_names
        }
        self._mode = default_mode
        # Services belonging to the arm that actually performs the wrist_3
        # rotation in ROTATE_CW / ROTATE_CCW. Every other arm is frozen (IDLE)
        # while rotating and resumes NORMAL follow once the mode returns to
        # NORMAL. Empty => legacy behaviour (all arms rotate together).
        self._rotation_services = set(rotation_services or [])
        if service_names:
            logging.info("GELLO control_mode services: %s", service_names)
            if self._rotation_services:
                frozen = [n for n in service_names if n not in self._rotation_services]
                logging.info(
                    "GELLO rotation restricted to %s; freezing %s during ROTATE_CW/ROTATE_CCW",
                    sorted(self._rotation_services),
                    frozen,
                )
        else:
            logging.warning("No GELLO control_mode services configured; mode changes are no-ops")

    @property
    def mode(self) -> int:
        return self._mode

    @staticmethod
    def mode_label(mode: int) -> str:
        if mode == GelloControlModeClient.MODE_IDLE:
            return "IDLE"
        if mode == GelloControlModeClient.MODE_NORMAL:
            return "NORMAL"
        if mode == GelloControlModeClient.MODE_CLOCKWISE:
            return "ROTATE_CW"
        if mode == GelloControlModeClient.MODE_COUNTERCLOCKWISE:
            return "ROTATE_CCW"
        return f"UNKNOWN({mode})"

    @classmethod
    def cycle_mode(cls, current_mode: int) -> int:
        if current_mode == cls.MODE_NORMAL:
            return cls.MODE_CLOCKWISE
        if current_mode == cls.MODE_CLOCKWISE:
            return cls.MODE_COUNTERCLOCKWISE
        return cls.MODE_NORMAL

    def _target_modes(self, mode: int) -> Dict[str, int]:
        """Per-service control_mode to request for the given logical mode.

        For NORMAL / IDLE every arm gets the same mode. For the rotation modes
        (CW / CCW), only the configured rotation arm(s) rotate; every other arm
        is frozen (IDLE) so it holds all joints — including wrist_3 — in place.
        """
        is_rotation = mode in (self.MODE_CLOCKWISE, self.MODE_COUNTERCLOCKWISE)
        if is_rotation and self._rotation_services:
            return {
                name: (mode if name in self._rotation_services else self.MODE_IDLE)
                for name in self._clients
            }
        return {name: mode for name in self._clients}

    def _apply_targets(self, targets: Dict[str, int], label: str) -> bool:
        """Write one ``control_mode`` value per service; True when all succeeded."""
        all_ok = True
        for name, client in self._clients.items():
            svc_mode = targets[name]
            request = SetParametersSrv.Request()
            request.parameters = [
                Parameter(
                    name="control_mode",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_INTEGER,
                        integer_value=int(svc_mode),
                    ),
                )
            ]

            if not client.service_is_ready():
                logging.info("Waiting for service %s...", name)
                client.wait_for_service()

            future = client.call_async(request)
            deadline = time.monotonic() + 10.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)

            if not future.done() or future.result() is None:
                logging.warning("[%s] %s: control_mode set timed out", label, name)
                all_ok = False
                continue

            resp = future.result()
            if not all(result.successful for result in resp.results):
                reasons = [result.reason for result in resp.results if not result.successful]
                logging.warning("[%s] %s: control_mode rejected: %s", label, name, reasons)
                all_ok = False

        return all_ok

    def set_mode(self, mode: int, label: str = "") -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported control mode: {mode}")

        if not self._clients:
            self._mode = mode
            logging.warning("[%s] control_mode -> %s (no services configured)", label, self.mode_label(mode))
            return

        targets = self._target_modes(mode)
        for name, svc_mode in targets.items():
            if svc_mode != mode:
                logging.info(
                    "[%s] %s: freezing at %s while %s active",
                    label, name, self.mode_label(svc_mode), self.mode_label(mode),
                )

        if self._apply_targets(targets, label):
            logging.info("[%s] control_mode -> %s (%d)", label, self.mode_label(mode), mode)
        else:
            logging.warning(
                "[%s] Some control_mode updates failed; current requested mode is %s (%d)",
                label,
                self.mode_label(mode),
                mode,
            )
        self._mode = mode

    def service_for_arm(self, arm_name: str) -> Optional[str]:
        """control_mode service whose path contains ``arm_name``.

        Same substring match :func:`resolve_rotation_services` uses, so ``left``
        selects ``/left_gello_offset_node/set_parameters``.
        """
        token = str(arm_name).strip().lower()
        if not token:
            return None
        for name in self._clients:
            if token in name.lower():
                return name
        return None

    def set_arm_modes(self, arm_modes: Dict[str, int], label: str = "") -> None:
        """Request a different ``control_mode`` per arm in one pass.

        ``arm_modes`` maps recorded arm keys ("left"/"right") to modes. Any
        service not named in ``arm_modes`` is set to IDLE, so a caller can never
        leave an unlisted arm following Gello by omission. Unresolvable arm
        names are skipped with a warning rather than silently ignored.
        """
        for mode in arm_modes.values():
            if mode not in self._VALID_MODES:
                raise ValueError(f"Unsupported control mode: {mode}")

        if not self._clients:
            logging.warning(
                "[%s] per-arm control_mode -> %s (no services configured)",
                label,
                {arm: self.mode_label(mode) for arm, mode in sorted(arm_modes.items())},
            )
            return

        targets = {name: self.MODE_IDLE for name in self._clients}
        for arm_name, mode in arm_modes.items():
            service = self.service_for_arm(arm_name)
            if service is None:
                logging.warning(
                    "[%s] no control_mode service matches arm %r (have %s); skipping it",
                    label, arm_name, sorted(self._clients),
                )
                continue
            targets[service] = mode

        all_ok = self._apply_targets(targets, label)
        summary = ", ".join(
            f"{name}={self.mode_label(mode)}" for name, mode in sorted(targets.items())
        )
        if all_ok:
            logging.info("[%s] per-arm control_mode -> %s", label, summary)
        else:
            logging.warning(
                "[%s] Some per-arm control_mode updates failed; requested %s", label, summary
            )

        # There is no single logical mode any more, but the overlay and the
        # cycle-mode handler both read ``mode``. NORMAL whenever some arm is
        # following keeps both of them honest.
        self._mode = (
            self.MODE_NORMAL
            if any(mode == self.MODE_NORMAL for mode in targets.values())
            else self.MODE_IDLE
        )


def resolve_control_mode_services(services_cfg: dict) -> List[str]:
    configured = services_cfg.get("control_mode") or []
    return list(configured)


def resolve_rotation_arm(cfg: dict) -> Optional[str]:
    """Arm name that performs wrist_3 rotation in ROTATE_CW/ROTATE_CCW.

    Returns ``None`` (rotate every arm together — legacy behaviour) when the
    top-level ``rotation_arm`` key is absent, null, or blank.
    """
    value = cfg.get("rotation_arm")
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def resolve_rotation_services(
    rotation_arm: Optional[str],
    control_mode_services: List[str],
) -> List[str]:
    """control_mode services whose arm should keep rotating in CW/CCW modes.

    Matches the rotation arm name against each service path (e.g. ``left``
    matches ``/left_gello_offset_node/set_parameters``). An empty result means
    every arm rotates together.
    """
    if not rotation_arm:
        return []
    token = rotation_arm.lower()
    matched = [s for s in control_mode_services if token in s.lower()]
    if not matched:
        logging.warning(
            "rotation_arm=%r did not match any control_mode service (%s); "
            "all arms will rotate together",
            rotation_arm, control_mode_services,
        )
    return matched


class SubtaskSpec(NamedTuple):
    """One subtask of a long-horizon episode: which arm drives it, and its name.

    ``name`` is cosmetic but load-bearing for the operator: "step 3/6 solid" can
    be checked against the bench at a glance, where a bare "step 3/6" cannot.
    """

    arm: str
    name: str = ""


def resolve_subtasks(cfg: dict) -> List[SubtaskSpec]:
    """Parse the optional top-level ``subtask_arms`` list into specs, in order.

    Each entry is either a bare arm name::

        subtask_arms: [left, left, right]

    or a mapping that also names the subtask::

        subtask_arms:
          - {arm: left,  name: stir bar}
          - {arm: right, name: solid}

    An empty result means "every arm follows Gello at all times", i.e. the
    behaviour of every recording made before subtask labelling existed.
    """
    value = cfg.get("subtask_arms")
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"subtask_arms must be a list, got {type(value).__name__}"
        )

    specs: List[SubtaskSpec] = []
    for position, entry in enumerate(value):
        if isinstance(entry, dict):
            unknown = sorted(set(entry) - {"arm", "name"})
            if unknown:
                raise ValueError(
                    f"subtask_arms[{position}] has unknown key(s) {unknown}; "
                    "expected 'arm' and optionally 'name'"
                )
            raw_arm = entry.get("arm")
            if raw_arm is None:
                raise ValueError(f"subtask_arms[{position}] is missing 'arm'")
            arm = str(raw_arm).strip().lower()
            name = str(entry.get("name") or "").strip()
        else:
            arm = str(entry).strip().lower()
            name = ""
        if not arm:
            raise ValueError(f"subtask_arms[{position}] has an empty arm name")
        specs.append(SubtaskSpec(arm=arm, name=name))
    return specs


def resolve_subtask_arms(cfg: dict) -> List[str]:
    """Acting arm per subtask; see :func:`resolve_subtasks`."""
    return [spec.arm for spec in resolve_subtasks(cfg)]


def resolve_subtask_names(cfg: dict) -> List[str]:
    """Display name per subtask, empty string where the config gave none."""
    return [spec.name for spec in resolve_subtasks(cfg)]


def subtask_label(
    subtask_names: List[str],
    subtask_index: int,
    subtask_count: int,
    acting_arm: Optional[str] = None,
) -> str:
    """Operator-facing ``step 3/6 solid (right)``, 1-based for display."""
    text = f"step {subtask_index + 1}/{subtask_count}"
    if 0 <= subtask_index < len(subtask_names) and subtask_names[subtask_index]:
        text += f" {subtask_names[subtask_index]}"
    if acting_arm:
        text += f" ({acting_arm})"
    return text


def validate_subtask_arms(subtask_arms: List[str], arm_keys: List[str]) -> None:
    """Raise when the arm map names an arm this session is not recording."""
    unknown = sorted(set(subtask_arms) - set(arm_keys))
    if unknown:
        raise ValueError(
            f"subtask_arms names arm(s) {unknown} that this session is not recording "
            f"(recording {list(arm_keys)}). Fix subtask_arms in the config, or adjust "
            "--left/--right so every mapped arm is included."
        )


def acting_arm_for_subtask(subtask_arms: List[str], subtask_index: int) -> Optional[str]:
    """Arm that follows Gello during ``subtask_index``, or None when unmapped."""
    if not subtask_arms:
        return None
    if subtask_index < 0 or subtask_index >= len(subtask_arms):
        return None
    return subtask_arms[subtask_index]


def arm_modes_for_subtask(
    arm_keys: List[str],
    subtask_arms: List[str],
    subtask_index: int,
) -> Dict[str, int]:
    """Per-arm control_mode for a subtask: acting arm NORMAL, every other IDLE.

    Returns an empty dict when ``subtask_index`` has no mapping, which callers
    treat as "leave the current modes alone".
    """
    acting = acting_arm_for_subtask(subtask_arms, subtask_index)
    if acting is None:
        return {}
    return {
        arm: (
            GelloControlModeClient.MODE_NORMAL
            if arm == acting
            else GelloControlModeClient.MODE_IDLE
        )
        for arm in arm_keys
    }


def resolve_transition_ready_services(services_cfg: dict, control_mode_services: List[str]) -> List[str]:
    configured = services_cfg.get("transition_ready")
    if configured is not None:
        return list(configured)

    derived: List[str] = []
    for service_name in control_mode_services:
        if service_name.endswith("/set_parameters"):
            derived.append(service_name[: -len("/set_parameters")] + "/transition_ready")
    return derived


def resolve_reset_services(services_cfg: dict) -> List[str]:
    configured = services_cfg.get("reset")
    if configured is None:
        return []
    return list(configured)


class GelloTransitionReadyClient:
    """Calls an empty readiness service once a Gello mode transition has been requested."""

    def __init__(self, node: "Node", service_names: List[str]) -> None:
        self._clients = {
            name: node.create_client(EmptySrv, name)
            for name in service_names
        }
        if service_names:
            logging.info("GELLO transition-ready services: %s", service_names)
        else:
            logging.warning("No GELLO transition-ready services configured; mode transitions will not wait for resume")

    def wait_for_resume(self, label: str = "") -> bool:
        if not self._clients:
            return True

        request = EmptySrv.Request()
        for name, client in self._clients.items():
            if not client.service_is_ready():
                logging.info("Waiting for transition-ready service %s...", name)
                client.wait_for_service()

            logging.info("[%s] Waiting for GELLO to resume via %s", label, name)
            future = client.call_async(request)
            while not future.done():
                time.sleep(0.05)

            if future.result() is None:
                logging.warning("[%s] %s: transition-ready service returned no response", label, name)
                return False

            logging.info("[%s] %s: GELLO resumed publishing", label, name)

        return True


class ResetServiceClient:
    """Calls optional reset services after the robot has been sent home."""

    def __init__(self, node: "Node", service_names: List[str]) -> None:
        self._clients = {
            name: node.create_client(EmptySrv, name)
            for name in service_names
        }
        if service_names:
            logging.info("Reset services: %s", service_names)
        else:
            logging.info("No reset services configured; reset will only home the robot")

    def call(self, label: str = "") -> bool:
        if not self._clients:
            return True

        request = EmptySrv.Request()
        all_ok = True
        for name, client in self._clients.items():
            if not client.service_is_ready():
                logging.info("Waiting for reset service %s...", name)
                client.wait_for_service()

            logging.info("[%s] Calling reset service %s", label, name)
            future = client.call_async(request)
            deadline = time.monotonic() + 60.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)

            if not future.done() or future.result() is None:
                logging.warning("[%s] %s: reset service call timed out", label, name)
                all_ok = False
                continue

            logging.info("[%s] %s: reset service completed", label, name)

        return all_ok


class ResetRequestClient:
    """Calls the external /reset Empty service to initiate a coordinated reset."""

    def __init__(self, node: "Node", service_names: List[str]) -> None:
        self._clients = {
            name: node.create_client(EmptySrv, name)
            for name in service_names
        }
        if service_names:
            logging.info("External reset request service(s): %s", service_names)
        else:
            logging.warning("No external reset request service configured; reset will be local only")

    def call(self, label: str = "") -> bool:
        if not self._clients:
            return True

        request = EmptySrv.Request()
        all_ok = True
        for name, client in self._clients.items():
            if not client.service_is_ready():
                logging.info("Waiting for reset request service %s...", name)
                client.wait_for_service()

            logging.info("[%s] Calling reset request service %s", label, name)
            future = client.call_async(request)
            deadline = time.monotonic() + 60.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)

            if not future.done() or future.result() is None:
                logging.warning("[%s] %s: reset request timed out", label, name)
                all_ok = False
                continue

            logging.info("[%s] %s: reset request completed", label, name)

        return all_ok


def start_recording_session(
    recording_state: RecordingState,
    control_mode_client: GelloControlModeClient,
    stop_event: threading.Event,
    label: str,
    arm_keys: Optional[List[str]] = None,
    subtask_arms: Optional[List[str]] = None,
    home_joint_names: Optional[dict[str, List[str]]] = None,
    home_sender: Optional[RobotHomeSender] = None,
    state_source: Optional[Callable[[str], Optional["np.ndarray"]]] = None,
) -> bool:
    if recording_state.is_recording:
        logging.warning("[%s] Recording is already active", label)
        return False

    # Claim the recording start FIRST (this honours the debounce). Only switch
    # GELLO to NORMAL if we actually started — otherwise a debounced or
    # duplicate press (e.g. a composite pedal echoing the same key on several
    # evdev nodes) would flip GELLO live while nothing is being recorded,
    # leaving the rig in "NORMAL but not recording".
    if not recording_state.start(label=label):
        logging.info("[%s] Start ignored (debounced); GELLO left unchanged", label)
        return False
    recording_state.clear_transient_label()

    # recording_state.start() reset the counter, so this is subtask 1's map.
    arm_modes = arm_modes_for_subtask(
        arm_keys or [], subtask_arms or [], recording_state.subtask_index
    )
    if not arm_modes:
        control_mode_client.set_mode(GelloControlModeClient.MODE_NORMAL, label=label)
        return True

    # Pin the arms that sit out subtask 1 before handing Gello to the acting
    # arm. Beyond holding them still, this seeds their action topic so the
    # recording loop will write frames at all — see
    # _publish_hold_at_current_state.
    if home_sender is not None and home_joint_names is not None:
        for arm_name, mode in sorted(arm_modes.items()):
            if mode != GelloControlModeClient.MODE_IDLE:
                continue
            _publish_hold_at_current_state(
                home_joint_names,
                home_sender,
                state_source,
                arm_name,
                label,
                message="Idle for subtask 1; holding at current state",
            )
    control_mode_client.set_arm_modes(arm_modes, label=label)
    return True


# Tolerances for the skip-if-already-home / verify-reached-home logic in the
# reset path. Values are in radians on non-gripper joints. Kept in sync with
# dagger.py, which uses the identical guard.
#   * SKIP threshold: tight — only skip the home publish when the arm is
#     essentially already there. Republishing a zero-delta home trajectory is
#     wasted wall-clock at best and, on the custom UR bridge, can leave the
#     controller in a state where the *next* non-zero command is ignored too
#     (the operator then has to restart the controller and recording).
#   * VERIFY threshold: looser — after send_home's per-arm settle sleep the arm
#     should be within this much of home; if not, the controller didn't act.
_HOME_SKIP_TOL_RAD = 0.02
_HOME_VERIFY_TOL_RAD = 0.08


def _arm_gripper_index(
    arm_configs: dict,
    home_joint_names: dict[str, List[str]],
    arm_name: str,
) -> int:
    """Index of the gripper joint in ``home_joint_names[arm_name]``, or -1."""
    cfg = arm_configs.get(arm_name)
    names = home_joint_names.get(arm_name, [])
    if cfg is None or not names:
        return -1
    suffix = cfg.gripper_joint
    for i, n in enumerate(names):
        if n.endswith(suffix):
            return i
    return -1


def _arm_non_gripper_max_delta(
    arm_configs: dict,
    home_positions: dict[str, List[float]],
    home_joint_names: dict[str, List[str]],
    state_source: Optional[Callable[[str], Optional["np.ndarray"]]],
    arm_name: str,
) -> Optional[float]:
    """Max-abs error (rad) between current arm state and the home target across
    all NON-gripper joints. ``None`` if state or config is missing."""
    names = home_joint_names.get(arm_name, [])
    home = home_positions.get(arm_name, [])
    if not names or not home or len(home) < len(names):
        return None
    current = state_source(arm_name) if state_source is not None else None
    if current is None or len(current) < len(names):
        return None
    g_idx = _arm_gripper_index(arm_configs, home_joint_names, arm_name)
    max_delta = 0.0
    for i in range(len(names)):
        if i == g_idx or i >= len(current):
            continue
        max_delta = max(max_delta, abs(float(current[i]) - float(home[i])))
    return max_delta


def _publish_hold_at_current_state(
    home_joint_names: dict[str, List[str]],
    home_sender: RobotHomeSender,
    state_source: Optional[Callable[[str], Optional["np.ndarray"]]],
    arm_name: str,
    label: str,
    overrides: Optional[Dict[int, float]] = None,
    message: str = "Hold-at-current-state published",
) -> bool:
    """Command ``arm_name`` to hold its current joint angles; True if published.

    ``overrides`` replaces individual joint targets by index, which is how the
    gripper-open pre-step moves one joint while pinning the rest.

    Publishing this before an arm is parked at ``control_mode = IDLE`` does two
    jobs. It holds the follower where it is, and it puts one message on the
    arm's action topic — the only way ``ROSArmStateListener._latest_action``
    ever becomes non-None. Without it, ``recording_loop``'s ``all_ready`` check
    never passes for an arm that has been IDLE since startup (nobody publishes
    for an idle arm) and the episode silently saves zero frames.
    """
    names = home_joint_names.get(arm_name, [])
    if not names:
        return False

    current = state_source(arm_name) if state_source is not None else None
    if current is None or len(current) < len(names):
        logging.warning("[%s] current state unavailable; cannot hold arm in place", arm_name)
        return False

    positions = [float(current[i]) for i in range(len(names))]
    for index, value in (overrides or {}).items():
        positions[index] = float(value)

    # Reuse the JointState publisher home_sender already owns so we don't race
    # with it on the same action topic.
    pub = home_sender._publishers.get(arm_name)
    if pub is None:
        return False

    msg = JointStateMsg()
    msg.header.stamp.sec = 3
    msg.header.stamp.nanosec = 0
    msg.name = list(names)
    msg.position = positions
    pub.publish(msg)
    logging.info("[%s] [%s] %s", label, arm_name, message)
    return True


def _open_gripper_before_home(
    arm_configs: dict,
    home_positions: dict[str, List[float]],
    home_joint_names: dict[str, List[str]],
    home_sender: RobotHomeSender,
    state_source: Optional[Callable[[str], Optional["np.ndarray"]]],
    arm_name: str,
    label: str,
) -> None:
    """Publish a JointState that holds the arm joints at their current state but
    moves the gripper joint to its home value.

    First step of the reset sequence: it lets go of whatever the gripper is
    holding *before* the arm is told to move home. Without it, an arm closed
    around an object fights the home trajectory and can trigger a protective
    stop that needs a robot reboot.
    """
    cfg = arm_configs.get(arm_name)
    names = home_joint_names.get(arm_name, [])
    home = home_positions.get(arm_name, [])
    if cfg is None or not names or not home:
        return

    gripper_suffix = cfg.gripper_joint
    try:
        g_idx = next(i for i, n in enumerate(names) if n.endswith(gripper_suffix))
    except StopIteration:
        logging.warning(
            "[%s] gripper joint suffix %r not found in home_joint_names; "
            "skipping gripper-open pre-step",
            arm_name, gripper_suffix,
        )
        return

    published = _publish_hold_at_current_state(
        home_joint_names,
        home_sender,
        state_source,
        arm_name,
        label,
        overrides={g_idx: float(home[g_idx])},
        message=(
            f"Gripper-open published (idx {g_idx}, target {float(home[g_idx]):.4f}); "
            "holding arm at current state"
        ),
    )
    if published:
        time.sleep(3.0)


def reset_recording_session(
    recording_state: RecordingState,
    control_mode_client: GelloControlModeClient,
    home_sender: RobotHomeSender,
    home_positions: dict[str, List[float]],
    home_joint_names: dict[str, List[str]],
    reset_service_client: ResetServiceClient,
    label: str,
    arm_configs: Optional[dict] = None,
    state_source: Optional[Callable[[str], Optional["np.ndarray"]]] = None,
) -> bool:
    if recording_state.is_recording:
        logging.warning("[%s] Reset requested while recording; stop first", label)
        return False

    recording_state.set_resetting(True)
    try:
        if arm_configs is None or state_source is None:
            # Legacy fallback: no state source wired up, so we can't reason
            # about whether the arm is already home — publish unconditionally.
            home_sender.send_home(home_positions, home_joint_names)
            reset_service_client.call(label=label)
        else:
            # Robust reset sequence (mirrors dagger.py):
            #   1. open right gripper, then left gripper
            #   2. home only arms that aren't already essentially at home
            #   3. verify each arm ended up at home
            reset_arm_order = [a for a in ("right", "left") if a in home_positions]
            for arm in reset_arm_order:
                _open_gripper_before_home(
                    arm_configs,
                    home_positions,
                    home_joint_names,
                    home_sender,
                    state_source,
                    arm,
                    label=label,
                )

            # Skip the home publish for arms already essentially at home.
            # Republishing a zero-delta trajectory can put the custom UR
            # bridge into a state where the next non-zero command is ignored,
            # forcing a controller + recording restart.
            homing_arms: List[str] = []
            for arm in reset_arm_order:
                delta = _arm_non_gripper_max_delta(
                    arm_configs, home_positions, home_joint_names, state_source, arm
                )
                if delta is None:
                    # State unavailable — be safe and publish.
                    homing_arms.append(arm)
                    continue
                if delta < _HOME_SKIP_TOL_RAD:
                    logging.info(
                        "[%s] [%s] already at home (max_delta=%.4f rad); "
                        "skipping home publish",
                        label, arm, delta,
                    )
                    continue
                homing_arms.append(arm)

            if homing_arms:
                ordered_home_positions = {a: home_positions[a] for a in homing_arms}
                ordered_home_joint_names = {a: home_joint_names[a] for a in homing_arms}
                home_sender.send_home(ordered_home_positions, ordered_home_joint_names)

            reset_service_client.call(label=label)

            # Verify every arm ended up at home. If not, the controller didn't
            # act (protective stop, inactive controller, dead hardware
            # interface) — surface it so the operator knows pedal presses
            # won't recover it.
            any_stuck = False
            for arm in reset_arm_order:
                delta = _arm_non_gripper_max_delta(
                    arm_configs, home_positions, home_joint_names, state_source, arm
                )
                if delta is None:
                    logging.warning(
                        "[%s] [%s] post-reset verification skipped — state unavailable",
                        label, arm,
                    )
                    continue
                if delta > _HOME_VERIFY_TOL_RAD:
                    any_stuck = True
                    logging.error(
                        "[%s] [%s] DID NOT REACH HOME after reset "
                        "(max_delta=%.3f rad, tol=%.3f). The arm controller on "
                        "ur_robotiq likely did not execute the trajectory — "
                        "possible protective stop, inactive controller, or dead "
                        "hardware interface. Check the ur_robotiq container logs "
                        "and `ros2 control list_hardware_components`. Pedal "
                        "presses will NOT recover this; a driver restart is "
                        "required.",
                        label, arm, delta, _HOME_VERIFY_TOL_RAD,
                    )
                else:
                    logging.info(
                        "[%s] [%s] at home (max_delta=%.4f rad)", label, arm, delta
                    )
            if any_stuck:
                logging.error(
                    "[%s] Reset finished with at least one arm NOT at home. "
                    "Recording further episodes is unsafe until the arm "
                    "controller is recovered.",
                    label,
                )
    finally:
        recording_state.set_resetting(False)
    logging.info("[%s] Reset complete; recording remains stopped", label)
    return True


def stop_recording_session(
    recording_state: RecordingState,
    control_mode_client: GelloControlModeClient,
    label: str,
) -> bool:
    if not recording_state.is_recording:
        logging.warning("[%s] Recording is not active", label)
        return False

    recording_state.stop(label=label)
    control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=label)
    return True


def discard_recording_session(
    recording_state: RecordingState,
    control_mode_client: GelloControlModeClient,
    dataset: "LeRobotDataset",
    recording_stats: RecordingStats,
    stats_path: Path,
    recording_hz: float,
    label: str,
) -> bool:
    if not recording_state.is_recording:
        logging.warning("[%s] Recording is not active", label)
        return False

    recording_state.discard(label=label)
    control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=label)
    dataset.clear_episode_buffer()
    recording_stats.record_discard()
    recording_stats.write(stats_path, recording_hz)
    recording_state.set_transient_label("DISCARDED", duration_s=0.9)
    return True


def discard_or_reset_recording_session(
    recording_state: RecordingState,
    control_mode_client: GelloControlModeClient,
    home_sender: RobotHomeSender,
    home_positions: dict[str, List[float]],
    home_joint_names: dict[str, List[str]],
    reset_service_client: ResetServiceClient,
    dataset: "LeRobotDataset",
    recording_stats: RecordingStats,
    stats_path: Path,
    recording_hz: float,
    label: str,
    arm_configs: Optional[dict] = None,
    state_source: Optional[Callable[[str], Optional["np.ndarray"]]] = None,
) -> bool:
    if recording_state.is_recording:
        return discard_recording_session(
            recording_state,
            control_mode_client,
            dataset,
            recording_stats,
            stats_path,
            recording_hz,
            label=label,
        )
    return reset_recording_session(
        recording_state,
        control_mode_client,
        home_sender,
        home_positions,
        home_joint_names,
        reset_service_client,
        label=label,
        arm_configs=arm_configs,
        state_source=state_source,
    )


def cycle_recording_mode(control_mode_client: GelloControlModeClient, label: str) -> int:
    next_mode = GelloControlModeClient.cycle_mode(control_mode_client.mode)
    control_mode_client.set_mode(next_mode, label=label)
    return next_mode


# ── Recording thread ──────────────────────────────────────────────────────

# Features declared with shape (1,) that LeRobot stores as scalar columns.
_SCALAR_INT_COLUMNS = ("subtask_index",)


def _coerce_scalar_episode_columns(dataset: "LeRobotDataset") -> None:
    """Flatten buffered shape-(1,) int columns to Python ints before saving.

    Works around a LeRobot + numpy-2.x mismatch, the same one dagger.py handles
    for ``action_source``. ``get_hf_features_from_features`` special-cases a
    shape-(1,) feature as a scalar ``datasets.Value("int64")`` column, but
    ``add_frame`` validates each entry as an ndarray of shape (1,). At
    ``save_episode`` time ``Dataset.from_dict`` calls ``int(value)`` per row,
    which numpy 2.x refuses on a 1-D array with ``TypeError: only 0-dimensional
    arrays can be converted to Python scalars`` — so without this every episode
    fails to save.
    """
    writer = getattr(dataset, "writer", None)
    if writer is None:
        return
    episode_buffer = getattr(writer, "episode_buffer", None)
    if not episode_buffer:
        return
    for column in _SCALAR_INT_COLUMNS:
        if column in episode_buffer:
            episode_buffer[column] = [
                int(np.asarray(value).reshape(-1)[0]) for value in episode_buffer[column]
            ]


def _warn_on_incomplete_subtasks(recording_state: RecordingState) -> None:
    """Flag an episode that ended before the last subtask.

    A missed pedal press is invisible in the video and mislabels every frame
    after it, so surface it now, while re-recording the episode is still cheap.
    """
    subtask_index, subtask_count = recording_state.subtask_snapshot()
    if subtask_count <= 0 or subtask_index == subtask_count - 1:
        return
    logging.warning(
        "Episode %d ended on subtask %d/%d — a subtask advance was probably "
        "missed, which mislabels every frame after it. Consider discarding and "
        "re-recording this episode.",
        recording_state.episode_idx - 1, subtask_index + 1, subtask_count,
    )


def recording_loop(
    state_left: Optional[ArmState],
    state_right: Optional[ArmState],
    cameras: List[CameraReader],
    dataset: "LeRobotDataset",
    recording_stats: RecordingStats,
    stats_path: Path,
    task: str,
    hz: float,
    recording_state: RecordingState,
    stop_event: threading.Event,
    resolution: Optional[Tuple[int, int]] = None,
    stabilizer: Optional[CameraStabilizer] = None,
    wrap_manager: Optional[WrapJointManager] = None,
    camera_failure: Optional[threading.Event] = None,
) -> None:
    record_h, record_w = resolution if resolution else (None, None)
    period = 1.0 / hz
    frames_in_episode = 0
    n_cam = len(cameras)
    if stabilizer is None:
        stabilizer = CameraStabilizer([False] * n_cam, [20.0] * n_cam)
    if wrap_manager is None:
        wrap_manager = WrapJointManager([])

    arm_states_by_key: Dict[str, Optional[ArmState]] = {"left": state_left, "right": state_right}

    # Live freeze watchdog: track per-camera stale state so the operator gets a
    # loud, rate-limited alert if any feed stops changing mid-episode. This also
    # covers the case where a camera's read thread is fully hung inside
    # cap.read() (it can't log for itself then), and counts how many frozen
    # frames were written so a bad episode can be spotted and re-done.
    cam_frozen_state: List[bool] = [False] * len(cameras)
    cam_frozen_frames: List[int] = [0] * len(cameras)
    freeze_threshold_s = max(0.5, 3.0 / hz)

    while not stop_event.is_set():
        t_start = time.monotonic()

        # A camera that can never be reopened (unplugged/crashed) would otherwise
        # keep the loop alive writing duplicate frames indefinitely. Bail out and
        # let main()'s teardown save the in-progress episode and finalize.
        dead_cam = next((c for c in cameras if c.is_unrecoverable()), None)
        if dead_cam is not None:
            logging.error(
                "Camera %s (%s) is unrecoverable — stopping recording and saving "
                "what we have.",
                dead_cam.name or dead_cam.display_label, dead_cam.device,
            )
            if camera_failure is not None:
                camera_failure.set()
            stop_event.set()
            break

        if recording_state.is_recording and not recording_state.is_capture_paused:
            snapshots: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
            all_ready = True
            for arm_key, arm_state in arm_states_by_key.items():
                if arm_state is None:
                    continue
                action, state, _ = arm_state.snapshot()
                if action is None:
                    all_ready = False
                snapshots[arm_key] = (action, state)

            if all_ready:
                if frames_in_episode == 0:
                    stabilizer.reset()
                    wrap_manager.reset_episode()
                    cam_frozen_frames = [0] * len(cameras)
                    cam_frozen_state = [False] * len(cameras)

                processed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for arm_key, (action, state) in snapshots.items():
                    state_mod, action_mod = wrap_manager.subtract_state_action(
                        arm_key, state, action, recording_state.episode_idx
                    )
                    processed[arm_key] = (state_mod, action_mod)

                action_parts, state_parts = [], []
                for arm_key in arm_states_by_key:
                    if arm_key not in processed:
                        continue
                    state_mod, action_mod = processed[arm_key]
                    action_parts.append(action_mod.astype(np.float32))
                    state_parts.append(state_mod.astype(np.float32))

                frame: dict = {
                    "task": task,
                    "action": np.concatenate(action_parts),
                    "observation.state": np.concatenate(state_parts),
                }
                subtask_index, subtask_count = recording_state.subtask_snapshot()
                if subtask_count > 0:
                    frame["subtask_index"] = np.asarray([subtask_index], dtype=np.int64)
                raw_frames = [cam.get_frame() for cam in cameras]
                for ci, cam in enumerate(cameras):
                    stale_for = cam.seconds_since_change()
                    if stale_for >= freeze_threshold_s:
                        cam_frozen_frames[ci] += 1
                        if not cam_frozen_state[ci]:
                            cam_frozen_state[ci] = True
                            logging.warning(
                                "FROZEN CAMERA during recording: %s (%s) — no new frame "
                                "for %.1fs. Recorded frames are DUPLICATES; this episode "
                                "should likely be discarded and re-recorded.",
                                cam.name or cam.display_label, cam.device, stale_for,
                            )
                    elif cam_frozen_state[ci]:
                        cam_frozen_state[ci] = False
                        logging.warning(
                            "Camera %s (%s) recovered after ~%d frozen frame(s) this episode",
                            cam.name or cam.display_label, cam.device, cam_frozen_frames[ci],
                        )
                stabilized = stabilizer.process(raw_frames)
                for ci, cam in enumerate(cameras):
                    bgr = stabilized[ci]
                    if bgr is not None:
                        if record_h is not None:
                            bgr = cv2.resize(bgr, (record_w, record_h))
                        frame[cam.feature_key] = Image.fromarray(
                            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        )
                try:
                    dataset.add_frame(frame)
                    frames_in_episode += 1
                except Exception:
                    logging.exception("Failed to append frame to dataset; dropping current episode buffer")
                    dataset.clear_episode_buffer()
                    frames_in_episode = 0
                    recording_state.discard(label="recorder")

        if recording_state.take_save_request():
            _warn_on_incomplete_subtasks(recording_state)
            if frames_in_episode > 0:
                try:
                    _coerce_scalar_episode_columns(dataset)
                    dataset.save_episode()
                    recording_stats.record_save()
                    logging.info(
                        f"Episode {recording_state.episode_idx - 1} saved "
                        f"({frames_in_episode} frames, {frames_in_episode / hz:.1f}s)"
                    )
                except Exception:
                    logging.exception(
                        "Failed to save episode %d; dropping buffered data and continuing",
                        recording_state.episode_idx - 1,
                    )
                    dataset.clear_episode_buffer()
                    recording_stats.record_discard()
                finally:
                    recording_stats.write(stats_path, hz)
            else:
                logging.warning("Empty episode discarded")
                recording_stats.record_discard()
                recording_stats.write(stats_path, hz)
            frames_in_episode = 0

        elapsed = time.monotonic() - t_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── Dataset features ──────────────────────────────────────────────────────

def build_features(
    arm_labels: List[str],
    arm_joint_names: Dict[str, List[str]],
    cameras: List[CameraReader],
    resolution: Optional[Tuple[int, int]] = None,
    subtask_count: int = 0,
) -> dict:
    joint_names = []
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
    }
    if subtask_count > 0:
        # 0-based phase label for long-horizon episodes, mirroring dagger.py's
        # action_source declaration. Shape (1,) is what LeRobot wants for a
        # scalar column; see _coerce_scalar_episode_columns for the catch.
        features["subtask_index"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["subtask_index"],
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
        description="Record monolateral or bimanual demonstrations via ROS 2 in LeRobot format."
    )
    parser.add_argument("--name", required=True,
                        help="Dataset name. Data saved to data/<name>/.")
    parser.add_argument("--task", required=True,
                        help='Task description, e.g. "pick up the block".')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--left", action="store_true", help="Left arm only.")
    group.add_argument("--right", action="store_true", help="Right arm only.")
    parser.add_argument("--hz", type=float,
                        default=None,
                        help="Recording frequency in Hz. "
                             "Defaults to recording.hz from --config.")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds (works with keyboard and/or pedal).")
    parser.add_argument(
        "--recording-control",
        type=str,
        default="keyboard",
        help=(
            "How to control recording start/stop.  "
            "Comma-separated list of: keyboard, pedal.  "
            "Example: --recording-control keyboard,pedal"
        ),
    )
    parser.add_argument(
        "--pedal-device",
        type=str,
        default=PEDAL_DEFAULT_DEVICE,
        help="evdev device path for the foot pedal (default: %(default)s).",
    )
    parser.add_argument(
        "--pedal-right-action",
        choices=("cycle-mode", "subtask"),
        default="cycle-mode",
        help="What the RIGHT foot pedal does. 'cycle-mode' (default) cycles "
             "GELLO NORMAL -> ROTATE_CW -> ROTATE_CCW. 'subtask' instead "
             "advances the per-frame subtask_index label for long-horizon "
             "episodes and hands GELLO to the arm listed for that subtask in "
             "the config's subtask_arms (every other arm is held IDLE). "
             "Keyboard 'm' still cycles mode either way, and in subtask mode "
             "'n'/'b' step the label forward/back.",
    )
    parser.add_argument(
        "--subtask-count",
        type=int,
        default=None,
        help="Number of subtasks per episode. Defaults to the length of "
             "subtask_arms in --config. Only used with "
             "--pedal-right-action subtask.",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Record at HxW resolution. Default: native camera resolution.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the named camera / teleop config file. "
                             "Defaults to $GELLO_CONFIG.")
    orient_group = parser.add_mutually_exclusive_group()
    orient_group.add_argument(
        "--stabilize-orientation",
        action="store_true",
        help="Enable 180° orientation stabilization for every camera (overrides config).",
    )
    orient_group.add_argument(
        "--no-stabilize-orientation",
        action="store_true",
        help="Disable orientation stabilization for every camera (overrides config).",
    )
    parser.add_argument(
        "--orientation-mae-delta-threshold",
        type=float,
        default=None,
        metavar="MAE",
        help="Override MAE gap (normal vs flipped) for all cameras; default from config or 20.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Mirror the recorded dataset to Hugging Face when done "
             "(default: no upload; or set LEROBOT_HF_PUSH=1).",
    )
    return parser.parse_args()


# ── Dataset-safety helpers ─────────────────────────────────────────────────

# Files a brand-new dataset writes before a single episode is recorded. Every
# other file under a dataset directory means real work lives there.
_SKELETON_RELATIVE_PATHS = {
    Path("meta/info.json"),
    Path("meta/recording_stats.json"),
    Path("recording_config.yaml"),
    Path("experiment_config.yaml"),
}

# Where a raw recording ends up once a dataset has been reorganized for
# training: the episodes are moved into one of these subdirectories while
# derived artifacts (train/, exports/, ...) sit alongside it. New episodes must
# be appended to that nested raw dataset, not to a fresh one at the top level.
_NESTED_RAW_DIR_NAMES = ("original_data", "raw", "recording", "recordings")

# Subdirectories that never hold a raw recording, so they are skipped when
# searching for the dataset to append to.
_NON_RAW_DIR_NAMES = frozenset({
    "data", "videos", "images", "meta",
    "train", "exports", "eval", "logs", "outputs", "checkpoints",
})


def _dataset_has_episode_data(root: Path) -> bool:
    """Return True if ``root`` holds anything beyond an empty dataset skeleton.

    Deliberately does not trust ``meta/info.json``: a session that crashed
    before ``finalize()`` leaves episodes on disk while ``info.json`` still
    reports ``total_episodes: 0``, and a dataset that was reorganized by hand
    (episodes moved under ``original_data/``, training variants under
    ``train/``) has no top-level ``data/`` or ``videos/`` at all. Both must be
    treated as irreplaceable.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root) not in _SKELETON_RELATIVE_PATHS:
            return True
    return False


def _dataset_episode_count(root: Path) -> int:
    """Episodes already recorded in the LeRobot dataset at ``root``.

    Returns 0 when ``root`` is not a LeRobot dataset or its metadata is
    unreadable.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return 0
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    try:
        return int(info.get("total_episodes", 0))
    except (TypeError, ValueError):
        return 0


def _durable_dataset_counters(root: Path) -> Optional[Tuple[int, int]]:
    """``(next_episode_index, next_frame_index)`` implied by the data on disk.

    Both counters in ``meta/info.json`` are allocators rather than
    descriptions: LeRobot stamps a new episode with ``total_episodes`` and its
    rows with ``arange(total_frames, ...)``, bumping both as soon as the
    episode is handed to the writer. Parquet footers only land when
    ``finalize()`` runs, so a session killed mid-run leaves both counters
    permanently ahead of the data that survived, and the next session starts
    past the gap where the lost episodes should have gone.

    Returns ``None`` when the surviving data cannot be read in full. Callers
    must keep the ``info.json`` values in that case -- a partial read
    understates the counters, and recording against an understated counter
    overwrites episodes that are still on disk.
    """
    episode_parts = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_parts = sorted((root / "data").rglob("*.parquet"))
    if not episode_parts or not data_parts:
        return None
    try:
        # Take the episode high-water mark across *both* the metadata and the
        # rows themselves. A session killed between its data flush and its
        # metadata write leaves an episode that exists only in ``data/``, and
        # allocating that index again would put two different episodes under
        # one number.
        last_episode = max(
            max(pq.read_table(p, columns=["episode_index"]).column("episode_index").to_pylist())
            for p in episode_parts + data_parts
        )
        last_frame = max(
            max(pq.read_table(p, columns=["index"]).column("index").to_pylist())
            for p in data_parts
        )
    except Exception:
        logging.exception(
            "Could not read the episode/frame indices under %s, so meta/info.json "
            "is being left exactly as it is", root,
        )
        return None
    return int(last_episode) + 1, int(last_frame) + 1


def _realign_counters_after_crash(root: Path) -> None:
    """Point ``info.json``'s allocators back at the data that actually landed.

    Runs before the dataset is resumed, because LeRobot reads both counters
    straight out of ``info.json`` to number the next episode. Skipped entirely
    unless every surviving parquet could be read, and the values written are
    one past the highest surviving index, so no counter can ever be moved to
    where an existing episode already sits.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return
    durable = _durable_dataset_counters(root)
    if durable is None:
        return
    next_episode, next_frame = durable
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    stale_episodes = int(info.get("total_episodes", 0))
    stale_frames = int(info.get("total_frames", 0))
    if (stale_episodes, stale_frames) == (next_episode, next_frame):
        return

    if stale_episodes > next_episode or stale_frames > next_frame:
        cause = (
            "a previous session was killed after its counters advanced but "
            "before its parquet footers landed, so the episodes it lost would "
            "be skipped over instead of refilled"
        )
    else:
        cause = (
            "a previous session wrote episode rows that its metadata never "
            "recorded, so the next episode would land on an index that "
            "already has frames"
        )
    logging.warning(
        "meta/info.json allocates episode %d / frame %d next, but the data on "
        "disk ends at episode %d / frame %d -- %s. Realigning to episode %d / "
        "frame %d.",
        stale_episodes, stale_frames, next_episode - 1, next_frame - 1, cause,
        next_episode, next_frame,
    )
    info["total_episodes"] = next_episode
    info["total_frames"] = next_frame
    info["splits"] = {"train": f"0:{next_episode}"}
    tmp_path = info_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(info, indent=4))
    os.replace(tmp_path, info_path)


def _find_resumable_dataset(root: Path) -> Optional[Path]:
    """Locate the dataset under ``root`` that new episodes should append to.

    Normally that is ``root`` itself. If ``root`` has been reorganized for
    training the raw episodes live one level down (``root/original_data`` by
    convention), so search there too instead of starting a new empty dataset
    beside them.

    Returns the directory to resume, or ``None`` when nothing resumable exists.
    """
    if _dataset_episode_count(root) > 0:
        return root
    if not root.is_dir():
        return None

    candidates = [root / name for name in _NESTED_RAW_DIR_NAMES]
    candidates += sorted(
        child for child in root.iterdir()
        if child.is_dir()
        and child.name not in _NON_RAW_DIR_NAMES
        and ".bak-" not in child.name
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        if _dataset_episode_count(candidate) > 0:
            return candidate
    return None


def _find_experiment_config(dataset_root: Path, requested_root: Path) -> Path:
    """Path of the experiment config (home positions) governing ``dataset_root``.

    Prefers the copy stored with the episodes. When a dataset has been
    reorganized the episodes move into a subdirectory while the config stays
    behind, so also look at the ancestors up to ``requested_root``: appending
    episodes recorded against freshly derived home positions would silently make
    the dataset inconsistent.

    Falls back to the path inside ``dataset_root`` so a first-time run creates it
    next to the episodes.
    """
    candidates = [dataset_root]
    current = dataset_root
    while current != requested_root and current.parent != current:
        current = current.parent
        candidates.append(current)
        if current == requested_root:
            break

    for directory in candidates:
        candidate = directory / "experiment_config.yaml"
        if candidate.exists():
            return candidate
    return dataset_root / "experiment_config.yaml"


def _resolve_dataset_root(root: Path) -> Tuple[Path, bool]:
    """Return the dataset directory to record into and whether it is resumed.

    Appending to existing episodes is the default: recording the same ``--name``
    twice adds to the dataset rather than replacing it. Only when no resumable
    dataset exists is a fresh one created, and then any leftover directory is
    handed to :func:`_resolve_existing_dataset_dir`, which never destroys data
    without an explicit confirmation.
    """
    resumable = _find_resumable_dataset(root)
    if resumable is not None:
        if resumable != root:
            logging.info(
                "Found %d existing episode(s) in %s; appending new episodes "
                "there (%s holds the reorganized copy).",
                _dataset_episode_count(resumable), resumable, root,
            )
        return resumable, True

    if root.exists():
        _resolve_existing_dataset_dir(root)
    return root, False


def _describe_dataset_dir(root: Path) -> str:
    """Human-readable one-line summary of what lives under ``root``."""
    n_files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            n_files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    mb = total_bytes / (1024 * 1024)
    return f"{n_files} file(s), {mb:.1f} MB"


def _timestamped_backup_path(root: Path) -> Path:
    """A sibling path like ``<root>.bak-YYYYmmdd-HHMMSS`` that does not exist."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = root.with_name(f"{root.name}.bak-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = root.with_name(f"{root.name}.bak-{stamp}-{suffix}")
        suffix += 1
    return candidate


def _resolve_existing_dataset_dir(root: Path) -> None:
    """Make ``root`` safe for a fresh ``LeRobotDataset.create`` WITHOUT ever
    destroying data behind the user's back.

    Called only when ``root`` exists but cannot be resumed. Guarantees:
      * Real recorded episodes are NEVER deleted without an explicit, typed
        confirmation from the user.
      * The default / non-interactive action is non-destructive: the existing
        folder is moved aside to a timestamped backup, never overwritten.

    Raises ``SystemExit`` if the user chooses to abort.
    """
    import shutil

    has_data = _dataset_has_episode_data(root)
    summary = _describe_dataset_dir(root)
    interactive = sys.stdin.isatty()

    if not has_data:
        # Only a skeleton (meta/ + config, no episodes). Nothing recorded is at
        # risk, but we still refuse to overwrite: move it aside so the new run
        # starts clean while the old skeleton is preserved.
        backup = _timestamped_backup_path(root)
        shutil.move(str(root), str(backup))
        logging.warning(
            "Existing dataset at %s had no recorded episodes (%s); moved it "
            "aside to %s and starting fresh (nothing was deleted).",
            root, summary, backup,
        )
        return

    # ── There IS real data here but no dataset we can append to. ──
    logging.error(
        "Dataset directory %s is not empty (%s) but holds no dataset with "
        "recorded episodes to append to — no meta/info.json reporting "
        "total_episodes > 0, here or in a nested %s directory. Likely a session "
        "that crashed before saving its first episode.",
        root, summary, "/".join(_NESTED_RAW_DIR_NAMES),
    )

    if not interactive:
        # Never delete data in a non-interactive context. Preserve everything
        # by moving it aside, then start fresh.
        backup = _timestamped_backup_path(root)
        shutil.move(str(root), str(backup))
        logging.warning(
            "Non-interactive session: preserved the existing data by moving it "
            "to %s instead of deleting it. Starting a fresh dataset at %s. "
            "Inspect the backup to recover those episodes.",
            backup, root,
        )
        return

    print()
    print("=" * 72)
    print(f"  EXISTING RECORDED DATA FOUND AT: {root}")
    print(f"  Contents: {summary}")
    print("  This data cannot be auto-resumed. Choose what to do:")
    print()
    print("    [k] KEEP it — move aside to a timestamped backup, start fresh")
    print("        (recommended, non-destructive; default)")
    print("    [d] DELETE it permanently, then start fresh")
    print("    [a] ABORT (do nothing)")
    print("=" * 72)

    while True:
        try:
            choice = input("Your choice [k/d/a] (default: k): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Aborted; existing data left untouched.")

        if choice in ("", "k", "keep"):
            backup = _timestamped_backup_path(root)
            shutil.move(str(root), str(backup))
            logging.warning("Moved existing data to %s; starting fresh at %s.", backup, root)
            return

        if choice in ("d", "delete"):
            confirm = input(
                f"Type the dataset name '{root.name}' to permanently delete it: "
            ).strip()
            if confirm != root.name:
                print("Names did not match; nothing deleted. Choose again.")
                continue
            shutil.rmtree(root)
            logging.warning("Permanently deleted %s at user's request.", root)
            return

        if choice in ("a", "abort", "q", "quit"):
            sys.exit("Aborted; existing data left untouched.")

        print("Unrecognized choice. Enter 'k', 'd', or 'a'.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not ROS_AVAILABLE:
        sys.exit("rclpy not found. Source your ROS 2 workspace first:\n  source /opt/ros/humble/setup.bash")
    if not LEROBOT_AVAILABLE:
        sys.exit("LeRobot not found. Install: pip install -e lerobot/")

    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)

    if args.hz is None:
        args.hz = float(cfg.get("recording", {}).get("hz", 10.0))

    if args.left:
        arm_keys = ["left"]
    elif args.right:
        arm_keys = ["right"]
    else:
        arm_keys = ["left", "right"]

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

    # ── Subtask labelling ────────────────────────────────────────────────
    subtask_mode = args.pedal_right_action == "subtask"
    try:
        subtask_specs = resolve_subtasks(cfg)
    except ValueError as exc:
        sys.exit(f"Invalid subtask_arms in {config_path}: {exc}")
    subtask_arms = [spec.arm for spec in subtask_specs]
    subtask_names = [spec.name for spec in subtask_specs]

    if not subtask_mode:
        # Keep every subtask code path inert, including the dataset column,
        # so a cycle-mode session behaves exactly as it did before.
        subtask_arms = []
        subtask_names = []
        subtask_count = 0
    else:
        if args.subtask_count is not None:
            subtask_count = int(args.subtask_count)
        elif subtask_arms:
            subtask_count = len(subtask_arms)
        else:
            sys.exit(
                "--pedal-right-action subtask needs to know how many subtasks an "
                f"episode has: add a subtask_arms list to {config_path}, or pass "
                "--subtask-count N."
            )
        if subtask_count < 1:
            sys.exit(f"--subtask-count must be at least 1, got {subtask_count}")
        if subtask_arms and len(subtask_arms) != subtask_count:
            sys.exit(
                f"--subtask-count {subtask_count} disagrees with the "
                f"{len(subtask_arms)} entries in subtask_arms ({config_path}); the "
                "label range and the arm map must describe the same subtasks."
            )
        try:
            validate_subtask_arms(subtask_arms, arm_keys)
        except ValueError as exc:
            sys.exit(str(exc))
        logging.info(
            "Subtask labelling ON: %d subtasks — %s",
            subtask_count,
            ", ".join(
                subtask_label(
                    subtask_names, i, subtask_count, acting_arm_for_subtask(subtask_arms, i)
                )
                for i in range(subtask_count)
            ),
        )

    try:
        camera_configs = load_camera_configs(cfg, selected_arms=arm_keys)
        camera_grid_columns = load_camera_grid_columns(cfg)
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

    if any(stabilize_flags):
        logging.info(
            "Camera orientation stabilization (180° vs previous frame): %s",
            ", ".join(
                f"{camera_configs[i].name}(Δ>{orientation_thresholds[i]:.1f})"
                for i in range(len(camera_configs))
                if stabilize_flags[i]
            ),
        )

    if args.resolution is not None:
        logging.info(f"Overriding camera resolution to {args.resolution[0]}x{args.resolution[1]}")

    # ── Cameras ──────────────────────────────────────────────────────────
    logging.info("Opening configured cameras...")
    try:
        cameras = open_configured_cameras(camera_configs, override_resolution=args.resolution)
    except RuntimeError as exc:
        sys.exit(str(exc))
    time.sleep(0.3)
    logging.info(f"{len(cameras)} camera(s) active")

    # The wrist cameras auto-rotate their own image when their gravity sensor
    # decides they are upside down, and stay that way for the rest of the
    # session. Number keys toggle a manual 180° correction so an inverted feed
    # can be put back the right way up without replugging the camera. The
    # recording thread and the preview each build their own CameraStabilizer,
    # so both are given this one shared object: a toggle from the preview has
    # to move the recorded frames too, or the operator would straighten the
    # display while the dataset kept the inverted feed.
    manual_flip = ManualFlip180(len(camera_configs))

    def _toggle_camera_flip(slot: int) -> None:
        new_state = manual_flip.toggle(slot)
        if new_state is None:
            return
        cam = cameras[slot]
        logging.warning(
            "[%s] manual 180° flip %s — recorded frames are now %s",
            cam.name or cam.display_label,
            "ON" if new_state else "OFF",
            "rotated" if new_state else "as-captured",
        )

    arm_labels = arm_keys  # used for feature naming

    repo_id = f"ur_robotiq/{args.name}"
    n_threads = max(4 * len(cameras), 4)

    requested_root = Path("data") / args.name
    root, _can_resume = _resolve_dataset_root(requested_root)
    stats_path = root / "meta" / "recording_stats.json"

    recording_stats = RecordingStats.load(stats_path) if _can_resume else RecordingStats()

    # ── ROS 2 node + listeners ───────────────────────────────────────────
    # Disable rclpy's built-in SIGINT handler so it can't race our own
    # ``_shutdown`` handler. We need our handler to win deterministically so
    # the ``finally`` block always runs ``dataset.finalize()`` before any ROS
    # teardown, and so a second Ctrl-C reliably escalates to ``os._exit``.
    if _RCLPY_SIGNAL_HANDLER_OPTIONS_AVAILABLE:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    else:
        rclpy.init()
    node = rclpy.create_node("imitation_recorder")

    svc_cfg = cfg.get("services", {})
    control_mode_services = resolve_control_mode_services(svc_cfg)
    transition_ready_services = resolve_transition_ready_services(svc_cfg, control_mode_services)
    reset_service_names = resolve_reset_services(svc_cfg)
    rotation_arm = resolve_rotation_arm(cfg)
    rotation_services = resolve_rotation_services(rotation_arm, control_mode_services)
    control_mode_client = GelloControlModeClient(
        node, control_mode_services, rotation_services=rotation_services
    )
    transition_ready_client = GelloTransitionReadyClient(node, transition_ready_services)
    reset_service_client = ResetServiceClient(node, reset_service_names)
    reset_request_client = ResetRequestClient(node, ["/reset"]) if use_reset_client else None
    recording_state = RecordingState(subtask_count=subtask_count)
    home_positions: dict[str, List[float]] = {}
    home_joint_names: dict[str, List[str]] = {}

    arm_states: Dict[str, ArmState] = {}
    listeners: Dict[str, ROSArmStateListener] = {}

    for key in arm_keys:
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
        # Must return LIVE state, not the first snapshot: RobotHomeSender now
        # polls this per waypoint to confirm the wrist settled before sending
        # the next target, so a stale initial reading would deadlock the
        # settle-check until the timeout fires.
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

    reset_return_service = None
    if use_reset_client:

        def _handle_reset_return(request: object, response: object):
            success = reset_recording_session(
                recording_state,
                control_mode_client,
                home_sender,
                home_positions,
                home_joint_names,
                reset_service_client,
                label="reset_return",
                arm_configs=arm_configs,
                state_source=_home_state_source,
            )
            response.success = bool(success)
            response.message = "reset completed" if success else "reset failed"
            return response

        reset_return_service = node.create_service(TriggerSrv, "/reset_return", _handle_reset_return)
        logging.info("Reset return service available at /reset_return")

    # Spin ROS in background using an explicit executor we own. Using a bare
    # ``rclpy.spin(node)`` leaves the wait-set owned by an anonymous internal
    # executor, which makes ``node.destroy_node()`` deadlock at shutdown:
    # destroy waits for spin to release the node, spin is parked on the
    # wait-set. With our own executor we can call ``executor.shutdown(timeout)``
    # from ``finally`` to wake the spin loop deterministically.
    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(node)

    def _spin_executor() -> None:
        try:
            ros_executor.spin()
        except Exception as exc:  # pragma: no cover - swallow on teardown
            logging.debug("ROS executor spin exited: %s", exc)

    spin_thread = threading.Thread(target=_spin_executor, daemon=True, name="ros_spin")
    spin_thread.start()

    # ── Shared state ─────────────────────────────────────────────────────
    stop_event = threading.Event()
    # Set by the recording thread when a camera dies and cannot be reopened;
    # lets the visualizer flash a big "CAM STUCK" banner before we exit+save.
    camera_failure = threading.Event()

    # ── Shutdown handler ─────────────────────────────────────────────────
    # The lock prevents two signals delivered in quick succession (e.g. the
    # user holding Ctrl-C) from racing through the ``_shutting_down`` check
    # before the first invocation has flipped the flag.
    _shutdown_lock = threading.Lock()
    _shutting_down = False
    _force_count = 0

    def _shutdown(sig: int, frame: object) -> None:
        nonlocal _shutting_down, _force_count
        with _shutdown_lock:
            if _shutting_down:
                _force_count += 1
                if _force_count >= 2:
                    logging.warning("Force exit (data may be incomplete)")
                    os._exit(1)
                return
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
        # Start in idle mode after topics are confirmed live.
        control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label="init")

    dataset = None

    if stop_event.is_set():
        logging.warning("Shutdown before initialization")
    else:
        experiment_path = _find_experiment_config(root, requested_root)
        experiment_config = load_experiment_home_config(experiment_path)
        should_save_experiment_config = experiment_config is None

        def _listener_state(arm_name: str) -> Tuple[List[str], List[float]]:
            listener = listeners.get(arm_name)
            if listener is None:
                return [], []
            return listener.initial_state()

        home_positions, home_joint_names = resolve_home(
            experiment_config, arm_keys, _listener_state
        )

        wrap_manager = WrapJointManager(wrap_joint_suffixes)
        wrap_manager.configure({arm: home_joint_names.get(arm, []) for arm in arm_keys})

        features = build_features(
            arm_labels,
            home_joint_names,
            cameras,
            resolution=args.resolution,
            subtask_count=subtask_count,
        )
        if _can_resume:
            logging.info(f"Resuming existing dataset at {root}")
            _realign_counters_after_crash(root)
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=root,
                image_writer_threads=n_threads,
            )
            # A resumed dataset keeps the schema it was created with, so a
            # subtask-mode run appending to a pre-subtask dataset (or the
            # reverse) would produce episodes whose label column disagrees with
            # the rest of the dataset. Refuse instead of silently splitting it.
            existing_features = set(getattr(dataset.meta, "features", {}) or {})
            if subtask_count > 0 and "subtask_index" not in existing_features:
                sys.exit(
                    f"{root} was recorded without a 'subtask_index' column, so "
                    "subtask-labelled episodes cannot be appended to it. Record into "
                    "a fresh --name, or backfill the column onto this dataset first "
                    "(src/lerobot_ros2/cli/add_action_source_to_base.py does the same "
                    "job for action_source and is the pattern to copy)."
                )
            if subtask_count == 0 and "subtask_index" in existing_features:
                sys.exit(
                    f"{root} has a 'subtask_index' column, but this run is not in "
                    "subtask mode so every new frame would be missing its label. "
                    "Re-run with --pedal-right-action subtask."
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
        # The pedal mapping is a launch-time flag, so record it alongside the
        # config: after the fact there is no other way to tell whether a
        # dataset's right-pedal presses meant "rotate" or "next subtask".
        resolved_cfg["recording"] = {
            **(resolved_cfg.get("recording") or {}),
            "pedal_right_action": args.pedal_right_action,
            # subtask_names is what makes the stored index readable later; without
            # it, "subtask_index == 2" means nothing six months from now.
            **(
                {"subtask_count": subtask_count, "subtask_names": list(subtask_names)}
                if subtask_count > 0
                else {}
            ),
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
            experiment_config = load_experiment_home_config(experiment_path)

        _banner = f"\nRecorder ready — {len(arm_keys)} arm(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        if use_keyboard:
            _banner += (
                "  Keyboard: 's' start/stop | 'd' discard/reset | 'r' reset | 'm' cycle mode | 'q' quit\n"
            )
            if subtask_mode:
                _banner += "            'n' next subtask | 'b' previous subtask\n"
        if use_pedal:
            _banner += (
                "  Pedal:    LEFT start/stop | MIDDLE discard/reset | RIGHT "
                + ("next subtask\n" if subtask_mode else "cycle mode\n")
            )
        _banner += (
            f"  GELLO mode: {GelloControlModeClient.mode_label(control_mode_client.mode)} ({control_mode_client.mode})\n"
        )
        if subtask_mode:
            _banner += f"  Subtasks: episodes start on step 1/{subtask_count}; press RIGHT to advance.\n"
            for i in range(subtask_count):
                _banner += (
                    "            "
                    + subtask_label(
                        subtask_names, i, subtask_count, acting_arm_for_subtask(subtask_arms, i)
                    )
                    + "\n"
                )
            _banner += (
                "            The idle arm holds its pose — you can let go of that GELLO\n"
                "            leader, but park it near the follower before it goes live again.\n"
            )
        logging.info(_banner)
        logging.info(f"Dataset -> {root.resolve()}  "
                     f"(episodes so far: {dataset.meta.total_episodes})")
        logging.info(f"State/action dim: {len(features['action']['names'])}, "
                     f"cameras: {len(cameras)}, hz: {args.hz}")

        recording_state.episode_idx = dataset.meta.total_episodes
        command_executor = UserCommandExecutor()
        transition_lock = threading.Lock()
        transition_queue: Deque[Tuple[str, Callable[[], None]]] = deque()
        transition_worker_running = False

        def _submit_toggle(source: str) -> None:
            def _action() -> None:
                if not recording_state.is_recording:
                    start_recording_session(
                        recording_state,
                        control_mode_client,
                        stop_event,
                        label=source,
                        arm_keys=arm_keys,
                        subtask_arms=subtask_arms,
                        home_joint_names=home_joint_names,
                        home_sender=home_sender,
                        state_source=_home_state_source,
                    )
                else:
                    stop_recording_session(recording_state, control_mode_client, label=source)
            command_executor.submit(source, "toggle", _action)

        def _submit_external_reset(source: str) -> None:
            def _action() -> None:
                control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=source)
                reset_request_client.call(label=source)

            command_executor.submit(source, "reset_request", _action)

        def _submit_local_reset(source: str) -> None:
            def _action() -> None:
                control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label=source)
                reset_recording_session(
                    recording_state,
                    control_mode_client,
                    home_sender,
                    home_positions,
                    home_joint_names,
                    reset_service_client,
                    label=source,
                    arm_configs=arm_configs,
                    state_source=_home_state_source,
                )

            command_executor.submit(source, "reset", _action)

        def _submit_discard_or_reset(source: str) -> None:
            if recording_state.is_recording:
                command_executor.submit(
                    source,
                    "discard",
                    lambda: discard_recording_session(
                        recording_state,
                        control_mode_client,
                        dataset,
                        recording_stats,
                        stats_path,
                        args.hz,
                        label=source,
                    ),
                )
                return

            if use_reset_client and reset_request_client is not None:
                _submit_external_reset(source)
                return

            _submit_local_reset(source)

        def _submit_reset(source: str) -> None:
            if use_reset_client and reset_request_client is not None:
                _submit_external_reset(source)
                return

            _submit_local_reset(source)

        def _run_with_transition_pause(source: str, request_mode: Callable[[], None]) -> None:
            """Queue a GELLO mode change behind a capture pause + resume wait.

            Capture is paused so the arm-settling transient never reaches the
            dataset, ``request_mode`` asks for the new mode, and we block on the
            transition-ready service until GELLO publishes again. Requests are
            serialised through one worker so rapid presses can't interleave
            mode writes or unpause capture while a later change is still
            settling.
            """
            nonlocal transition_worker_running

            recording_state.set_transition_pending(True)
            if recording_state.is_recording:
                recording_state.set_capture_paused(True)

            with transition_lock:
                transition_queue.append((source, request_mode))
                should_start_worker = not transition_worker_running
                if should_start_worker:
                    transition_worker_running = True

            if should_start_worker:
                threading.Thread(
                    target=_drain_transition_queue,
                    name=f"gello-transition-{source}",
                    daemon=True,
                ).start()

        def _drain_transition_queue() -> None:
            nonlocal transition_worker_running

            transition_confirmed = True
            try:
                while True:
                    with transition_lock:
                        if not transition_queue:
                            break
                        source, request_mode = transition_queue.popleft()

                    request_mode()
                    transition_confirmed = transition_ready_client.wait_for_resume(label=source)
                    if not transition_confirmed:
                        logging.warning("[%s] Transition completion not confirmed; keeping WAITING state", source)
                        break
            finally:
                with transition_lock:
                    transition_worker_running = False
                    queued_left = len(transition_queue)

                if transition_confirmed and queued_left == 0:
                    recording_state.set_transition_pending(False)
                    recording_state.set_capture_paused(False)

        def _submit_cycle_mode(source: str) -> None:
            _run_with_transition_pause(
                source, lambda: cycle_recording_mode(control_mode_client, label=source)
            )

        def _submit_step_subtask(source: str, delta: int) -> None:
            def _action() -> None:
                previous_index = recording_state.subtask_index
                new_index = (
                    recording_state.advance_subtask(label=source)
                    if delta > 0
                    else recording_state.retreat_subtask(label=source)
                )
                if new_index is None:
                    return

                acting_before = acting_arm_for_subtask(subtask_arms, previous_index)
                acting_after = acting_arm_for_subtask(subtask_arms, new_index)
                logging.info(
                    "[%s] %s -> %s",
                    source,
                    subtask_label(subtask_names, previous_index, subtask_count, acting_before),
                    subtask_label(subtask_names, new_index, subtask_count, acting_after),
                )
                recording_state.set_subtask_flash(
                    subtask_label(subtask_names, new_index, subtask_count).upper(),
                    duration_s=1.5,
                )

                if acting_after == acting_before:
                    # The same arm keeps Gello, so this press is a pure label
                    # change: no service call, no capture pause, and therefore no
                    # discontinuity in the middle of the recorded trajectory.
                    return

                arm_modes = arm_modes_for_subtask(arm_keys, subtask_arms, new_index)
                if not arm_modes:
                    return
                # No hold publish needed here, unlike at episode start: IDLE
                # itself pins the outgoing arm, and its action topic has been
                # live for the frames it just drove.
                logging.info(
                    "[%s] Handing Gello: %s -> %s (pausing capture while it settles)",
                    source, acting_before or "all", acting_after,
                )
                _run_with_transition_pause(
                    source, lambda: control_mode_client.set_arm_modes(arm_modes, label=source)
                )

            command_executor.submit(source, f"subtask{delta:+d}", _action)

        # ── Recording thread ─────────────────────────────────────────────
        record_stabilizer = CameraStabilizer(
            stabilize_flags, orientation_thresholds, manual_flip=manual_flip
        )
        rec_thread = threading.Thread(
            target=recording_loop,
            args=(
                arm_states.get("left"),
                arm_states.get("right"),
                cameras,
                dataset,
                recording_stats,
                stats_path,
                args.task,
                args.hz,
                recording_state,
                stop_event,
                args.resolution,
                record_stabilizer,
                wrap_manager,
                camera_failure,
            ),
            name="recording",
            daemon=True,
        )
        rec_thread.start()

        # ── Foot-pedal thread ─────────────────────────────────────────────
        if use_pedal:
            pedal_thread = FootPedalThread(
                device_path=args.pedal_device,
                stop_event=stop_event,
                on_toggle=lambda: _submit_toggle("pedal"),
                on_discard=lambda: _submit_discard_or_reset("pedal"),
                on_reset=(
                    (lambda: _submit_step_subtask("pedal", 1))
                    if subtask_mode
                    else (lambda: _submit_cycle_mode("pedal"))
                ),
            )
            pedal_thread.start()

    # ── Main loop: optional OpenCV preview (any recording-control mode) + optional keyboard ──
    try:
        if dataset is not None and args.visualize and cameras:
            cam_labels = [c.display_label for c in cameras]
            if use_keyboard:
                win = "record.py — Q/ESC quit | S start/stop | D discard/reset | R reset | M cycle mode | 1-9 flip cam 180°"
                if subtask_mode:
                    win += " | N/B subtask"
            else:
                win = "record.py — live preview — Q/ESC quit (record via pedal)"

            viz_stabilizer = CameraStabilizer(
                stabilize_flags, orientation_thresholds, manual_flip=manual_flip
            )

            def _flip_annotated_labels() -> List[str]:
                return [
                    f"{label}  [FLIP180]" if manual_flip.is_flipped(i) else label
                    for i, label in enumerate(cam_labels)
                ]

            def _draw_record_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
                countdown_label, countdown_remaining = recording_state.get_countdown()
                stabilization_count = viz_stabilizer.active_count

                # Loud, centered alert for the teleoperator when a camera has
                # died and we're about to exit + save.
                if camera_failure.is_set():
                    dead = next((c for c in cameras if c.is_unrecoverable()), None)
                    banner = "CAM STUCK"
                    sub = "saving & exiting"
                    if dead is not None:
                        sub = f"{dead.name or dead.display_label} stuck - saving & exiting"
                    scale = disp_w / 320.0
                    (tw, th), _ = cv2.getTextSize(
                        banner, cv2.FONT_HERSHEY_SIMPLEX, scale, max(4, int(scale * 3))
                    )
                    bx = (disp_w - tw) // 2
                    by = (disp_h + th) // 2
                    cv2.putText(canvas, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 0, 0), max(8, int(scale * 6)))
                    cv2.putText(canvas, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 0, 255), max(4, int(scale * 3)))
                    (sw, _sh), _ = cv2.getTextSize(
                        sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                    )
                    sx = (disp_w - sw) // 2
                    sy = by + int(th * 0.9)
                    cv2.putText(canvas, sub, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 0), 4)
                    cv2.putText(canvas, sub, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 1)

                if recording_state.is_transition_pending:
                    label = f"WAITING  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 320, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 120), 4)
                    cv2.putText(canvas, label, (disp_w - 320, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                elif recording_state.is_recording:
                    label = f"  REC  ep {recording_state.episode_idx}"
                    subtask_index, live_subtask_count = recording_state.subtask_snapshot()
                    if live_subtask_count > 0:
                        label += "  " + subtask_label(
                            subtask_names,
                            subtask_index,
                            live_subtask_count,
                            acting_arm_for_subtask(subtask_arms, subtask_index) or "all",
                        )
                    # Right-align: the subtask suffix makes the label too wide
                    # for a fixed offset.
                    (label_w, _lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 4)
                    label_origin = (max(16, disp_w - label_w - 16), disp_h - 16)
                    cv2.putText(canvas, label, label_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 4)
                    cv2.putText(canvas, label, label_origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
                elif recording_state.is_resetting:
                    label = f"RESET  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 300, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 220), 4)
                    cv2.putText(canvas, label, (disp_w - 300, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                elif (transient := recording_state.get_transient_label()) is not None:
                    label = f"{transient}  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 340, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 4)
                    cv2.putText(canvas, label, (disp_w - 340, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 255), 2)
                elif countdown_label is not None:
                    seconds_left = max(1, math.ceil(countdown_remaining))
                    label = f"{countdown_label} {seconds_left}  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 380, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 120), 4)
                    cv2.putText(canvas, label, (disp_w - 380, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    label = f"IDLE  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 280, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(canvas, label, (disp_w - 280, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)

                # Drawn independently of the status-line branches above so it is
                # visible mid-episode, which is the only time it fires. Large and
                # centered because the operator glances up for a fraction of a
                # second while both hands are on the GELLO leaders.
                if (flash := recording_state.get_subtask_flash()) is not None:
                    flash_scale = max(0.9, disp_w / 640.0)
                    flash_thick = max(3, int(flash_scale * 2))
                    (fw, fh), _ = cv2.getTextSize(
                        flash, cv2.FONT_HERSHEY_SIMPLEX, flash_scale, flash_thick
                    )
                    fx = max(8, (disp_w - fw) // 2)
                    fy = fh + 28
                    cv2.putText(canvas, flash, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX,
                                flash_scale, (0, 0, 0), flash_thick + 4)
                    cv2.putText(canvas, flash, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX,
                                flash_scale, (60, 255, 255), flash_thick)

                if stabilization_count:
                    stab_text = f"STAB ON ({stabilization_count}/{viz_stabilizer.num_cameras})"
                    cv2.putText(canvas, stab_text, (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(canvas, stab_text, (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

                flipped_names = [
                    cameras[i].display_label or f"cam{i + 1}"
                    for i, flipped in enumerate(manual_flip.flags())
                    if flipped and i < len(cameras)
                ]
                if flipped_names:
                    flip_text = "FLIP180: " + ", ".join(flipped_names)
                    flip_y = 64 if stabilization_count else 34
                    cv2.putText(canvas, flip_text, (16, flip_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(canvas, flip_text, (16, flip_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)

                mode_text = (
                    f"MODE {GelloControlModeClient.mode_label(control_mode_client.mode)} "
                    f"({control_mode_client.mode})"
                )
                mode_text += "  PEDAL SUBTASK" if subtask_mode else "  PEDAL MODE-CYCLE"
                if control_mode_client.mode == GelloControlModeClient.MODE_IDLE:
                    gello_dot_color = (0, 0, 255)
                elif recording_state.is_transition_pending:
                    gello_dot_color = (0, 255, 255)
                else:
                    gello_dot_color = (0, 200, 0)

                mode_origin = (16, disp_h - 16)
                dot_center = (mode_origin[0] + 12, mode_origin[1] - 12)
                mode_text_origin = (mode_origin[0] + 30, mode_origin[1])
                cv2.circle(canvas, dot_center, 8, (0, 0, 0), 3)
                cv2.circle(canvas, dot_center, 6, gello_dot_color, -1)
                cv2.putText(canvas, mode_text, mode_text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                cv2.putText(canvas, mode_text, mode_text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 230, 255), 1)

            with LivePreview(
                window_name=win,
                cam_labels=cam_labels,
                initial_size=(1280, 720),
                grid_columns=camera_grid_columns,
                wait_key_ms=30,
            ) as preview:
                while not stop_event.is_set():
                    frames = viz_stabilizer.process([c.get_frame() for c in cameras])
                    preview.set_labels(_flip_annotated_labels())
                    key = preview.render(frames, _draw_record_overlay)

                    if key in (ord('q'), ord('Q'), 27):
                        stop_event.set()
                    elif use_keyboard:
                        if key in (ord('s'), ord('S')):
                            _submit_toggle("keyboard")
                        elif key in (ord('d'), ord('D')):
                            _submit_discard_or_reset("keyboard")
                        elif key in (ord('r'), ord('R')):
                            _submit_reset("keyboard")
                        elif key in (ord('m'), ord('M')):
                            _submit_cycle_mode("keyboard")
                        elif subtask_mode and key in (ord('n'), ord('N')):
                            _submit_step_subtask("keyboard", 1)
                        elif subtask_mode and key in (ord('b'), ord('B')):
                            _submit_step_subtask("keyboard", -1)
                        elif ord('1') <= key <= ord('9'):
                            _toggle_camera_flip(key - ord('1'))

                # A camera died: hold the big "CAM STUCK" banner on screen long
                # enough for the teleoperator to read it before we exit + save.
                if camera_failure.is_set():
                    dwell_until = time.monotonic() + 3.0
                    while time.monotonic() < dwell_until:
                        frames = viz_stabilizer.process(
                            [c.get_frame() for c in cameras]
                        )
                        preview.render(frames, _draw_record_overlay)
        elif dataset is not None and use_keyboard:
            keyboard_handlers: Dict[str, Callable[[], None]] = {
                "s": lambda: _submit_toggle("keyboard"),
                "r": lambda: _submit_reset("keyboard"),
                "d": lambda: _submit_discard_or_reset("keyboard"),
                "m": lambda: _submit_cycle_mode("keyboard"),
            }
            if subtask_mode:
                keyboard_handlers["n"] = lambda: _submit_step_subtask("keyboard", 1)
                keyboard_handlers["b"] = lambda: _submit_step_subtask("keyboard", -1)
            for _slot in range(min(len(cameras), 9)):
                keyboard_handlers[str(_slot + 1)] = (
                    lambda slot=_slot: _toggle_camera_flip(slot)
                )
            run_cbreak_keyboard_loop(stop_event, keyboard_handlers)
        elif dataset is not None:
            _active = []
            if use_pedal:
                _ppaths = resolve_pedal_evdev_paths(args.pedal_device)
                _pnames = ", ".join(Path(p).name for p in _ppaths)
                _active.append(f"foot pedal ({len(_ppaths)} node(s): {_pnames})")
            logging.info(
                "Recording control via %s. Ctrl-C to quit.",
                " + ".join(_active) if _active else "Ctrl-C only",
            )
            while not stop_event.is_set():
                time.sleep(0.2)

    except Exception as e:
        logging.error(f"Main loop error: {e}")
        stop_event.set()
    finally:
        # ── Dataset durability (critical path) ─────────────────────────
        try:
            if dataset is not None and recording_state.is_recording:
                logging.info("Saving in-progress episode before exit...")
                _coerce_scalar_episode_columns(dataset)
                dataset.save_episode()
                recording_stats.record_save()
                recording_stats.write(stats_path, args.hz)
                recording_state.episode_idx += 1
        except Exception as e:
            logging.warning(f"Could not save in-progress episode: {e}")

        try:
            if dataset is not None:
                dataset.finalize()
                recording_stats.write(stats_path, args.hz)
                logging.info(
                    f"Dataset finalized — {recording_state.episode_idx} episode(s) "
                    f"-> {root.resolve()}"
                )
        except Exception as e:
            logging.warning(f"Error finalizing dataset: {e}")

        # ── Hardware + ROS teardown (best effort, watchdogged) ─────────
        # By here the dataset is on disk; everything below is camera/ROS
        # cleanup that has historically deadlocked (rclpy executor + spin
        # thread + destroy_node interaction). A watchdog timer guarantees
        # we always exit cleanly even if rclpy refuses to.
        def _force_exit_watchdog() -> None:
            logging.warning(
                "ROS/camera teardown exceeded 5s; forcing exit "
                "(dataset is already saved)"
            )
            os._exit(0)

        teardown_watchdog = threading.Timer(5.0, _force_exit_watchdog)
        teardown_watchdog.daemon = True
        teardown_watchdog.start()

        try:
            for cam in cameras:
                try:
                    cam.stop()
                except Exception:
                    pass

            # Best-effort: flip GELLO back to IDLE so the next lerobot-ros
            # run inherits a sane state. Does not gate shutdown if the
            # service is gone.
            try:
                control_mode_client.set_mode(
                    GelloControlModeClient.MODE_IDLE, label="shutdown"
                )
            except Exception as e:
                logging.warning(f"Could not reset GELLO to IDLE on shutdown: {e}")

            # Wake the spin thread *before* destroying the node so the
            # executor releases its wait-set and ``destroy_node`` doesn't
            # deadlock against an in-flight ``spin()``.
            try:
                ros_executor.shutdown(timeout_sec=2.0)
            except Exception as e:
                logging.debug("ROS executor shutdown raised: %s", e)
            spin_thread.join(timeout=2.0)

            try:
                node.destroy_node()
            except Exception as e:
                logging.debug("node.destroy_node raised: %s", e)
            try:
                rclpy.try_shutdown()
            except Exception as e:
                logging.debug("rclpy.try_shutdown raised: %s", e)

            logging.info("Shutdown complete")
        finally:
            teardown_watchdog.cancel()

        # ── Mirror the finished dataset to Hugging Face (best-effort) ──────
        # Runs after ROS/camera teardown so the upload can't interfere with
        # shutdown. Never raises; enable with --push or LEROBOT_HF_PUSH=1.
        try_sync_to_hub(root, push=args.push)


if __name__ == "__main__":
    main()
