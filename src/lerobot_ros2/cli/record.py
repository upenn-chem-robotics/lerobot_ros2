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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from PIL import Image
import yaml

os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)

from lerobot_ros2.config_paths import resolve_config_path
from lerobot_ros2.helper import (
    ArmState,
    CameraReader,
    CameraStabilizer,
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
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters as SetParametersSrv
    from std_srvs.srv import Empty as EmptySrv
    from std_srvs.srv import Trigger as TriggerSrv
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


def load_camera_grid_columns(cfg: dict) -> int:
    visualization_cfg = cfg.get("visualization") or {}
    return int(visualization_cfg.get("camera_grid_columns", 1))


class RecordingState:
    """Controls episode recording lifecycle, safe to call from multiple threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_recording = False
        self.episode_idx = 0
        self._save_requested = False
        self._last_toggle_time = 0.0
        self._is_resetting = False
        self._capture_paused = False
        self._transition_pending = False
        self._transient_label: Optional[str] = None
        self._transient_until = 0.0
        self._countdown_label: Optional[str] = None
        self._countdown_until = 0.0

    def toggle(self, label: str = "") -> Optional[str]:
        with self._lock:
            if self.is_recording:
                self._stop_locked(label=label)
                return "stopped"
            self._start_locked(label=label)
            return "started"

    def start(self, label: str = "") -> None:
        with self._lock:
            self._start_locked(label=label)

    def stop(self, label: str = "") -> None:
        with self._lock:
            self._stop_locked(label=label)

    def discard(self, label: str = "") -> None:
        with self._lock:
            self._stop_locked(label=label, save_episode=False)

    def _start_locked(self, label: str = "") -> None:
        now = time.monotonic()
        if now - self._last_toggle_time < 1.0:
            return
        self._last_toggle_time = now
        self.is_recording = True
        logging.info(f"[{label}] ● Recording STARTED — episode {self.episode_idx}")

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

    def __init__(self, node: "Node", service_names: List[str], default_mode: int = MODE_NORMAL) -> None:
        self._node = node
        self._clients = {
            name: node.create_client(SetParametersSrv, name)
            for name in service_names
        }
        self._mode = default_mode
        if service_names:
            logging.info("GELLO control_mode services: %s", service_names)
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

    def set_mode(self, mode: int, label: str = "") -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported control mode: {mode}")

        if not self._clients:
            self._mode = mode
            logging.warning("[%s] control_mode -> %s (no services configured)", label, self.mode_label(mode))
            return

        request = SetParametersSrv.Request()
        request.parameters = [
            Parameter(
                name="control_mode",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=int(mode),
                ),
            )
        ]

        all_ok = True
        for name, client in self._clients.items():
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

        if all_ok:
            self._mode = mode
            logging.info("[%s] control_mode -> %s (%d)", label, self.mode_label(mode), mode)
        else:
            logging.warning(
                "[%s] Some control_mode updates failed; current requested mode is %s (%d)",
                label,
                self.mode_label(mode),
                mode,
            )
            self._mode = mode


def resolve_control_mode_services(services_cfg: dict) -> List[str]:
    configured = services_cfg.get("control_mode") or []
    return list(configured)


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
) -> bool:
    if recording_state.is_recording:
        logging.warning("[%s] Recording is already active", label)
        return False

    control_mode_client.set_mode(GelloControlModeClient.MODE_NORMAL, label=label)
    recording_state.clear_transient_label()
    recording_state.start(label=label)
    return True


def reset_recording_session(
    recording_state: RecordingState,
    control_mode_client: GelloControlModeClient,
    home_sender: RobotHomeSender,
    home_positions: dict[str, List[float]],
    home_joint_names: dict[str, List[str]],
    reset_service_client: ResetServiceClient,
    label: str,
) -> bool:
    if recording_state.is_recording:
        logging.warning("[%s] Reset requested while recording; stop first", label)
        return False

    recording_state.set_resetting(True)
    try:
        home_sender.send_home(home_positions, home_joint_names)
        reset_service_client.call(label=label)
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
    )


def cycle_recording_mode(control_mode_client: GelloControlModeClient, label: str) -> int:
    next_mode = GelloControlModeClient.cycle_mode(control_mode_client.mode)
    control_mode_client.set_mode(next_mode, label=label)
    return next_mode


# ── Recording thread ──────────────────────────────────────────────────────

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

    while not stop_event.is_set():
        t_start = time.monotonic()

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
                raw_frames = [cam.get_frame() for cam in cameras]
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
            if frames_in_episode > 0:
                try:
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

    arm_labels = arm_keys  # used for feature naming
    import shutil

    root = Path("data") / args.name
    repo_id = f"ur_robotiq/{args.name}"
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
        logging.info(f"Removed incomplete dataset at {root}")

    recording_stats = RecordingStats.load(stats_path) if _can_resume else RecordingStats()

    # ── ROS 2 node + listeners ───────────────────────────────────────────
    rclpy.init()
    node = rclpy.create_node("imitation_recorder")

    svc_cfg = cfg.get("services", {})
    control_mode_services = resolve_control_mode_services(svc_cfg)
    transition_ready_services = resolve_transition_ready_services(svc_cfg, control_mode_services)
    reset_service_names = resolve_reset_services(svc_cfg)
    control_mode_client = GelloControlModeClient(node, control_mode_services)
    transition_ready_client = GelloTransitionReadyClient(node, transition_ready_services)
    reset_service_client = ResetServiceClient(node, reset_service_names)
    reset_request_client = ResetRequestClient(node, ["/reset"]) if use_reset_client else None
    recording_state = RecordingState()
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
            )
            response.success = bool(success)
            response.message = "reset completed" if success else "reset failed"
            return response

        reset_return_service = node.create_service(TriggerSrv, "/reset_return", _handle_reset_return)
        logging.info("Reset return service available at /reset_return")

    # Spin ROS in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Shared state ─────────────────────────────────────────────────────
    stop_event = threading.Event()

    # ── Shutdown handler ─────────────────────────────────────────────────
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
        # Start in idle mode after topics are confirmed live.
        control_mode_client.set_mode(GelloControlModeClient.MODE_IDLE, label="init")

    dataset = None

    if stop_event.is_set():
        logging.warning("Shutdown before initialization")
    else:
        experiment_path = root / "experiment_config.yaml"
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

        features = build_features(arm_labels, home_joint_names, cameras, resolution=args.resolution)
        if _can_resume:
            logging.info(f"Resuming existing dataset at {root}")
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
        if use_pedal:
            _banner += (
                "  Pedal:    LEFT start/stop | MIDDLE discard/reset | RIGHT cycle mode\n"
            )
        _banner += (
            f"  GELLO mode: {GelloControlModeClient.mode_label(control_mode_client.mode)} ({control_mode_client.mode})\n"
        )
        logging.info(_banner)
        logging.info(f"Dataset -> {root.resolve()}  "
                     f"(episodes so far: {dataset.meta.total_episodes})")
        logging.info(f"State/action dim: {len(features['action']['names'])}, "
                     f"cameras: {len(cameras)}, hz: {args.hz}")

        recording_state.episode_idx = dataset.meta.total_episodes
        command_executor = UserCommandExecutor()
        mode_cycle_lock = threading.Lock()
        mode_cycle_pending = 0
        mode_cycle_worker_running = False

        def _submit_toggle(source: str) -> None:
            def _action() -> None:
                if not recording_state.is_recording:
                    start_recording_session(recording_state, control_mode_client, stop_event, label=source)
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

        def _submit_cycle_mode(source: str) -> None:
            nonlocal mode_cycle_pending, mode_cycle_worker_running

            recording_state.set_transition_pending(True)
            if recording_state.is_recording:
                recording_state.set_capture_paused(True)

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

                    cycle_recording_mode(control_mode_client, label=source)
                    transition_confirmed = transition_ready_client.wait_for_resume(label=source)
                    if not transition_confirmed:
                        logging.warning("[%s] Transition completion not confirmed; keeping WAITING state", source)
                        break
            finally:
                with mode_cycle_lock:
                    mode_cycle_worker_running = False
                    queued_left = mode_cycle_pending

                if transition_confirmed and queued_left == 0:
                    recording_state.set_transition_pending(False)
                    recording_state.set_capture_paused(False)

        # ── Recording thread ─────────────────────────────────────────────
        record_stabilizer = CameraStabilizer(stabilize_flags, orientation_thresholds)
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
                on_reset=lambda: _submit_cycle_mode("pedal"),
            )
            pedal_thread.start()

    # ── Main loop: optional OpenCV preview (any recording-control mode) + optional keyboard ──
    try:
        if dataset is not None and args.visualize and cameras:
            cam_labels = [c.display_label for c in cameras]
            if use_keyboard:
                win = "record.py — Q/ESC quit | S start/stop | D discard/reset | R reset | M cycle mode"
            else:
                win = "record.py — live preview — Q/ESC quit (record via pedal)"

            viz_stabilizer = CameraStabilizer(stabilize_flags, orientation_thresholds)

            def _draw_record_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
                countdown_label, countdown_remaining = recording_state.get_countdown()
                stabilization_count = viz_stabilizer.active_count

                if recording_state.is_transition_pending:
                    label = f"WAITING  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 320, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 120), 4)
                    cv2.putText(canvas, label, (disp_w - 320, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                elif recording_state.is_recording:
                    label = f"  REC  ep {recording_state.episode_idx}"
                    cv2.putText(canvas, label, (disp_w - 300, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 4)
                    cv2.putText(canvas, label, (disp_w - 300, disp_h - 16),
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

                if stabilization_count:
                    stab_text = f"STAB ON ({stabilization_count}/{viz_stabilizer.num_cameras})"
                    cv2.putText(canvas, stab_text, (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(canvas, stab_text, (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

                mode_text = (
                    f"MODE {GelloControlModeClient.mode_label(control_mode_client.mode)} "
                    f"({control_mode_client.mode})"
                )
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
        elif dataset is not None and use_keyboard:
            run_cbreak_keyboard_loop(
                stop_event,
                {
                    "s": lambda: _submit_toggle("keyboard"),
                    "r": lambda: _submit_reset("keyboard"),
                    "d": lambda: _submit_discard_or_reset("keyboard"),
                    "m": lambda: _submit_cycle_mode("keyboard"),
                },
            )
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
        try:
            if dataset is not None and recording_state.is_recording:
                logging.info("Saving in-progress episode before exit...")
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
                logging.info(f"Done — {recording_state.episode_idx} episode(s) -> {root.resolve()}")
        except Exception as e:
            logging.warning(f"Error finalizing dataset: {e}")

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
            logging.warning(f"Could not reset GELLO to IDLE on shutdown: {e}")

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
