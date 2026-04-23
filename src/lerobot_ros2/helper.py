#!/usr/bin/env python3
"""Shared helpers for the LeRobot recorder and deploy scripts."""

import contextlib
import logging
import math
import os
import glob
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import cv2
import numpy as np
import yaml
import time

import select

try:
    from evdev import InputDevice, categorize, ecodes as ev_ecodes
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False

try:
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    JointStateMsg = object  # type: ignore[assignment]
    JointTrajectoryMsg = object  # type: ignore[assignment]


@contextlib.contextmanager
def quiet_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def load_config(config_path: Path) -> dict:
    """Load ``gello.yaml`` from ``config_path``.

    There is no default: CLIs are expected to resolve the path via
    ``lerobot_ros2.config_paths.resolve_config_path`` and pass it in.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def wrap_pi(x):
    """Fold an angle (radians) into the half-open interval (-pi, pi].

    Works element-wise on numpy arrays and on plain floats.
    """
    arr = np.asarray(x, dtype=np.float64)
    wrapped = np.mod(arr + np.pi, 2.0 * np.pi) - np.pi
    if np.isscalar(x):
        return float(wrapped)
    return wrapped


def find_wrap_indices(
    joint_names: Sequence[str],
    wrap_suffixes: Sequence[str] = ("wrist_3_joint",),
) -> List[int]:
    """Return indices of joints in ``joint_names`` whose tail matches any suffix.

    Matches are suffix-based so names like ``left_wrist_3_joint`` and
    ``right_wrist_3_joint`` both resolve to the generic ``wrist_3_joint`` entry
    in the config.
    """
    if not joint_names or not wrap_suffixes:
        return []
    indices: List[int] = []
    suffixes = tuple(wrap_suffixes)
    for i, name in enumerate(joint_names):
        if not isinstance(name, str):
            continue
        if name.endswith(suffixes):
            indices.append(i)
    return indices


def resolve_wrap_joints(cfg: dict) -> List[str]:
    """Read ``wrap_joints`` from the loaded gello.yaml dict, defaulting to wrist_3.

    Accepts either a list of strings or ``null``/missing (falls back to
    ``["wrist_3_joint"]``). An explicit empty list disables the feature.
    """
    raw = cfg.get("wrap_joints") if isinstance(cfg, dict) else None
    if raw is None:
        return ["wrist_3_joint"]
    if not isinstance(raw, list):
        raise ValueError("wrap_joints must be a list of joint-name suffixes (e.g. ['wrist_3_joint'])")
    out: List[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"wrap_joints entries must be non-empty strings, got {entry!r}")
        out.append(entry)
    return out


@dataclass(frozen=True)
class CameraConfig:
    index: int
    name: str
    port: str
    resolution: Optional[Tuple[int, int]] = None
    stabilize_orientation_180: bool = False
    orientation_mae_delta_threshold: float = 20.0
    settings: Dict[str, int] = field(default_factory=dict)

    @property
    def feature_key(self) -> str:
        return f"observation.images.camera_{self.index:02d}_{_slugify_camera_name(self.name)}"

    @property
    def display_label(self) -> str:
        return self.name


# Controls that gate other settings: they must be applied BEFORE the dependent
# absolute value, otherwise the absolute value is silently rejected by the
# driver (it stays in `flags=inactive`).
_AUTO_GATE_KEYS: Tuple[str, ...] = (
    "auto_exposure",
    "exposure_auto",
    "white_balance_automatic",
    "white_balance_temperature_auto",
    "focus_automatic_continuous",
    "focus_auto",
    "hue_auto",
    "gain_automatic",
)


class CameraSettingsMismatch(RuntimeError):
    """Raised when a v4l2 control we tried to set did not stick after read-back."""

    def __init__(self, port: str, diff: Dict[str, Tuple[Any, Any]]):
        self.port = port
        self.diff = diff
        parts = ", ".join(
            f"{key}: requested={requested!r} got={got!r}"
            for key, (requested, got) in diff.items()
        )
        super().__init__(f"Camera {port}: settings did not stick — {parts}")


def _order_settings_for_apply(settings: Dict[str, int]) -> List[Tuple[str, int]]:
    """Apply `auto_*` / `*_automatic` gate keys first, everything else after."""
    ordered: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for key in _AUTO_GATE_KEYS:
        if key in settings:
            ordered.append((key, int(settings[key])))
            seen.add(key)
    for key, value in settings.items():
        if key not in seen:
            ordered.append((key, int(value)))
    return ordered


_V4L2_GET_INT_RE = re.compile(r"-?\d+")


def _v4l2_get_ctrl(port: str, key: str) -> Optional[int]:
    """Return the current integer value of a v4l2 control, or None if unreadable.

    Menu controls on recent v4l2-utils versions print the current value
    followed by the human-readable label in parentheses, e.g.
    ``auto_exposure: 1 (Manual Mode)``. We extract the leading signed
    integer from the portion after the first ``:`` so both plain and
    labelled outputs parse correctly.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", port, f"--get-ctrl={key}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    if ":" not in out:
        return None
    raw = out.split(":", 1)[1].strip()
    match = _V4L2_GET_INT_RE.search(raw)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def apply_v4l2_settings(port: str, settings: Dict[str, int]) -> Dict[str, int]:
    """Apply `settings` to the camera at `port` via v4l2-ctl, then verify.

    Order is enforced so that `auto_*` / `*_automatic` gates are disabled
    BEFORE any dependent absolute value is written. Each control is read back
    after the batch; if any control didn't stick, ``CameraSettingsMismatch``
    is raised with the full diff so the caller can fail fast instead of
    silently recording on auto-drifting cameras.
    """
    if not settings:
        return {}

    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError(
            "v4l2-ctl not found on PATH; install it with `apt install v4l-utils` "
            "or remove the `settings:` blocks from gello.yaml."
        )

    ordered = _order_settings_for_apply(settings)
    for key, value in ordered:
        result = subprocess.run(
            ["v4l2-ctl", "--device", port, f"--set-ctrl={key}={value}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout).strip()
            raise CameraSettingsMismatch(
                port=port,
                diff={key: (value, f"set failed: {stderr}")},
            )

    applied: Dict[str, int] = {}
    diff: Dict[str, Tuple[Any, Any]] = {}
    for key, requested in settings.items():
        got = _v4l2_get_ctrl(port, key)
        applied[key] = got if got is not None else -1
        if got != int(requested):
            diff[key] = (int(requested), got)

    if diff:
        raise CameraSettingsMismatch(port=port, diff=diff)
    return applied


def _slugify_camera_name(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return slug.strip("_") or "camera"


def _parse_resolution(resolution: Any) -> Optional[Tuple[int, int]]:
    if resolution is None:
        return None
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("Camera resolution must be a 2-item [H, W] sequence")
    height, width = int(resolution[0]), int(resolution[1])
    if height <= 0 or width <= 0:
        raise ValueError("Camera resolution values must be positive")
    return height, width


def _orientation_options_from_entry(
    entry: dict,
    default_stabilize: bool,
    default_mae_delta: float,
) -> Tuple[bool, float]:
    if "stabilize_orientation_180" in entry:
        stabilize = bool(entry["stabilize_orientation_180"])
    else:
        stabilize = default_stabilize
    if "orientation_mae_delta_threshold" in entry:
        mae_delta = float(entry["orientation_mae_delta_threshold"])
    else:
        mae_delta = default_mae_delta
    return stabilize, mae_delta


def _settings_from_entry(
    entry: dict,
    default_settings: Dict[str, int],
) -> Dict[str, int]:
    """Merge `camera_defaults.settings` with this camera's `settings:` overrides.

    Per-camera overrides win on a per-key basis, so adding `focus_absolute: 100`
    on a wrist camera doesn't blow away the shared exposure/white-balance pin.
    """
    merged: Dict[str, int] = dict(default_settings)
    raw = entry.get("settings")
    if raw is None:
        return merged
    if not isinstance(raw, dict):
        raise ValueError("Camera `settings:` must be a mapping of v4l2 control names to integers")
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Camera setting keys must be non-empty strings")
        try:
            merged[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Camera setting {key!r} must be an integer, got {value!r}") from exc
    return merged


def _parse_default_settings(cfg: dict) -> Dict[str, int]:
    defaults_block = cfg.get("camera_defaults") or {}
    raw = defaults_block.get("settings") if isinstance(defaults_block, dict) else None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("`camera_defaults.settings` must be a mapping of v4l2 control names to integers")
    out: Dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError("camera_defaults.settings keys must be non-empty strings")
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera_defaults.settings[{key!r}] must be an integer, got {value!r}") from exc
    return out


def _parse_camera_entries(
    raw_cameras: Any,
    default_resolution: Optional[Tuple[int, int]],
    default_stabilize: bool,
    default_mae_delta: float,
    default_settings: Dict[str, int],
) -> List[Tuple[str, str, Optional[Tuple[int, int]], bool, float, Dict[str, int]]]:
    camera_entries: List[Tuple[str, str, Optional[Tuple[int, int]], bool, float, Dict[str, int]]] = []

    if isinstance(raw_cameras, dict):
        for name, entry in raw_cameras.items():
            camera_name = str(name).strip()
            if not camera_name:
                raise ValueError("Camera names must be non-empty")
            if not isinstance(entry, dict):
                raise ValueError(f"Camera {camera_name!r} must map to a configuration object")

            port = str(entry.get("port", "")).strip()
            if not port:
                raise ValueError(f"Camera {camera_name!r} is missing port")
            resolution = _parse_resolution(entry.get("resolution")) or default_resolution
            stabilize, mae_delta = _orientation_options_from_entry(
                entry, default_stabilize, default_mae_delta
            )
            settings = _settings_from_entry(entry, default_settings)
            camera_entries.append((camera_name, port, resolution, stabilize, mae_delta, settings))
        return camera_entries

    if isinstance(raw_cameras, list):
        for entry in raw_cameras:
            if not isinstance(entry, dict):
                raise ValueError("Each camera entry must be a mapping")

            name = str(entry.get("name", "")).strip()
            port = str(entry.get("port", "")).strip()
            if not name:
                raise ValueError("Camera entry is missing name")
            if not port:
                raise ValueError(f"Camera {name!r} is missing port")
            resolution = _parse_resolution(entry.get("resolution")) or default_resolution
            stabilize, mae_delta = _orientation_options_from_entry(
                entry, default_stabilize, default_mae_delta
            )
            settings = _settings_from_entry(entry, default_settings)
            camera_entries.append((name, port, resolution, stabilize, mae_delta, settings))
        return camera_entries

    raise ValueError("Camera sections must be mappings of camera names or lists of camera entries")


def _selected_camera_sections(selected_arms: Optional[Sequence[str]]) -> set[str]:
    if not selected_arms:
        return {"left_cameras", "right_cameras", "other_cameras"}

    selection = {str(arm).strip().lower() for arm in selected_arms if str(arm).strip()}
    sections = {"other_cameras"}
    if "left" in selection:
        sections.add("left_cameras")
    if "right" in selection:
        sections.add("right_cameras")
    return sections


def load_camera_configs(cfg: dict, selected_arms: Optional[Sequence[str]] = None) -> List[CameraConfig]:
    default_resolution = _parse_resolution(cfg.get("camera_resolution"))
    recording_cfg = cfg.get("recording") or {}
    default_stabilize = bool(recording_cfg.get("stabilize_orientation_180", False))
    default_mae_delta = float(recording_cfg.get("orientation_mae_delta_threshold", 20.0))
    default_settings = _parse_default_settings(cfg)

    camera_entries: List[Tuple[str, str, Optional[Tuple[int, int]], bool, float, Dict[str, int]]] = []
    grouped_sections = {
        section_name: cfg.get(section_name)
        for section_name in ("left_cameras", "right_cameras", "other_cameras")
        if cfg.get(section_name) is not None
    }

    if grouped_sections:
        selected_sections = _selected_camera_sections(selected_arms)
        for section_name in ("left_cameras", "right_cameras", "other_cameras"):
            raw_section = grouped_sections.get(section_name)
            if raw_section is None or section_name not in selected_sections:
                continue
            camera_entries.extend(
                _parse_camera_entries(
                    raw_section,
                    default_resolution,
                    default_stabilize,
                    default_mae_delta,
                    default_settings,
                )
            )
    else:
        raw_cameras = cfg.get("cameras") or {}
        camera_entries = _parse_camera_entries(
            raw_cameras,
            default_resolution,
            default_stabilize,
            default_mae_delta,
            default_settings,
        )

    if not camera_entries:
        raise ValueError(
            "No cameras configured. Add entries under `cameras:` or grouped `left_cameras`/"
            "`right_cameras`/`other_cameras` sections."
        )

    camera_entries.sort(key=lambda item: item[0].casefold())
    camera_configs: List[CameraConfig] = []
    seen_names = set()
    for index, (name, port, resolution, stabilize, mae_delta, settings) in enumerate(camera_entries):
        if name in seen_names:
            raise ValueError(f"Duplicate camera name {name!r}")
        seen_names.add(name)
        camera_configs.append(
            CameraConfig(
                index=index,
                name=name,
                port=port,
                resolution=resolution,
                stabilize_orientation_180=stabilize,
                orientation_mae_delta_threshold=mae_delta,
                settings=settings,
            )
        )

    return camera_configs


@dataclass(frozen=True)
class ArmTopicConfig:
    name: str
    action_topic: str
    state_topic: str
    tf_prefix: str = ""
    gripper_joint: str = "robotiq_85_left_knuckle_joint"


def load_arm_configs(cfg: dict) -> dict[str, ArmTopicConfig]:
    raw_arms = cfg.get("arms") or {}
    if not raw_arms:
        raise ValueError("No arm configs found. Add an `arms:` mapping with left/right entries.")

    arm_configs: dict[str, ArmTopicConfig] = {}
    for arm_name in ("left", "right"):
        entry = raw_arms.get(arm_name)
        if entry is None:
            raise ValueError(f"Missing arm config for {arm_name!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"Arm config for {arm_name!r} must be a mapping")

        action_topic = str(entry.get("action_topic", "")).strip()
        state_topic = str(entry.get("state_topic", "")).strip()
        if not action_topic:
            raise ValueError(f"Arm config {arm_name!r} is missing action_topic")
        if not state_topic:
            raise ValueError(f"Arm config {arm_name!r} is missing state_topic")

        tf_prefix = str(entry.get("tf_prefix", f"{arm_name}_")).strip()
        gripper_joint = str(entry.get("gripper_joint", "robotiq_85_left_knuckle_joint")).strip()

        arm_configs[arm_name] = ArmTopicConfig(
            name=arm_name,
            action_topic=action_topic,
            state_topic=state_topic,
            tf_prefix=tf_prefix,
            gripper_joint=gripper_joint,
        )

    return arm_configs


class ArmState:
    """Thread-safe snapshot of one arm pair's current state."""

    def __init__(self) -> None:
        self.action_qpos: Optional[np.ndarray] = None
        self.state_qpos: Optional[np.ndarray] = None
        self.synchronized: bool = False
        self._lock = threading.Lock()

    def update(self, action_qpos: np.ndarray, state_qpos: np.ndarray, synced: bool) -> None:
        with self._lock:
            self.action_qpos = action_qpos.copy()
            self.state_qpos = state_qpos.copy()
            self.synchronized = synced

    def snapshot(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        with self._lock:
            action_qpos = self.action_qpos.copy() if self.action_qpos is not None else None
            state_qpos = self.state_qpos.copy() if self.state_qpos is not None else None
            return action_qpos, state_qpos, self.synchronized


def orientation_stabilize_180_vs_prev(
    prev: Optional[np.ndarray],
    current: np.ndarray,
    mae_delta_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """If the feed suddenly matches the *previous* frame better when rotated 180°, use the rotation.

    Compares mean absolute difference to ``prev`` for ``current`` vs ``rotate180(current)``.
    When the flipped version wins by at least ``mae_delta_threshold``, returns the flipped frame.

    Returns ``(frame_to_use, new_prev)`` where ``new_prev`` is a copy of ``frame_to_use`` for the
    next call (temporal consistency).
    """
    if prev is None or prev.shape != current.shape or prev.dtype != current.dtype:
        return current, current.copy()

    flipped = cv2.rotate(current, cv2.ROTATE_180)
    diff_normal = cv2.absdiff(current, prev)
    error_normal = float(np.mean(diff_normal))
    diff_flipped = cv2.absdiff(flipped, prev)
    error_flipped = float(np.mean(diff_flipped))

    if error_flipped < error_normal and (error_normal - error_flipped) > mae_delta_threshold:
        out = flipped
    else:
        out = current
    return out, out.copy()


class CameraReader:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(
        self,
        device: str,
        width: int = 640,
        height: int = 360,
        index: Optional[int] = None,
        name: str = "",
        feature_key: str = "",
        display_label: str = "",
        settings: Optional[Dict[str, int]] = None,
    ) -> None:
        self.device = device
        self.index = index if index is not None else self._derive_index(device)
        self.name = name
        self.feature_key = feature_key
        self.display_label = display_label or f"cam {self.index}"
        self.requested_settings: Dict[str, int] = dict(settings or {})
        self.applied_settings: Dict[str, int] = {}
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Could not open camera at {device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self._cap.release()
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                self._cap.release()
                raise RuntimeError(f"Could not open camera at {device}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self._cap.release()
            raise RuntimeError(f"Camera at {device} did not report a valid resolution")

        if self.requested_settings:
            try:
                self.applied_settings = apply_v4l2_settings(device, self.requested_settings)
            except Exception:
                self._cap.release()
                raise
            logging.info(
                "Camera[%s] %s @ %s — applied v4l2 settings: %s",
                self.index,
                self.name or "(unnamed)",
                device,
                ", ".join(f"{k}={v}" for k, v in self.applied_settings.items()),
            )

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"cam_{self.index}"
        )

    @staticmethod
    def _derive_index(device: str) -> int:
        basename = os.path.basename(device)
        digits = "".join(ch for ch in basename if ch.isdigit())
        return int(digits) if digits else 0

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def detect_cameras() -> List[CameraReader]:
    by_path = sorted(glob.glob("/dev/v4l/by-path/*-video-index0"))
    if by_path:
        paths = [os.path.realpath(p) for p in by_path]
        logging.info(f"Using stable USB paths: {dict(zip(by_path, paths))}")
    else:
        paths = sorted(
            (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
            key=lambda p: int(p[len("/dev/video"):]),
        )

    readers: List[CameraReader] = []
    with quiet_stderr():
        for path in paths:
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if ret and w > 0 and h > 0:
                    reader = CameraReader(path)
                    reader.start()
                    readers.append(reader)
                    logging.info(f"Camera {path}: {reader.width}x{reader.height}")
            else:
                cap.release()
    if not readers:
        logging.warning("No cameras detected")
    return readers


def open_configured_cameras(
    camera_configs: List[CameraConfig],
    override_resolution: Optional[Tuple[int, int]] = None,
) -> List[CameraReader]:
    readers: List[CameraReader] = []
    with quiet_stderr():
        try:
            for camera in sorted(camera_configs, key=lambda item: item.index):
                resolution = override_resolution or camera.resolution
                width = resolution[1] if resolution else 640
                height = resolution[0] if resolution else 360
                reader = CameraReader(
                    camera.port,
                    width=width,
                    height=height,
                    index=camera.index,
                    name=camera.name,
                    feature_key=camera.feature_key,
                    display_label=camera.display_label,
                    settings=camera.settings,
                )
                reader.start()
                readers.append(reader)
                logging.info(
                    f"Camera[{camera.index}] {camera.name}: {camera.port} -> "
                    f"{reader.width}x{reader.height}"
                )
        except Exception:
            for reader in readers:
                try:
                    reader.stop()
                except Exception:
                    pass
            raise
    return readers


class ROSArmStateListener:
    """Subscribes to per-arm action and state JointState topics."""

    def __init__(
        self,
        node: "Node",
        action_topic: str,
        state_topic: str,
        arm_state: ArmState,
        label: str = "",
    ) -> None:
        self._arm_state = arm_state
        self._label = label

        self._latest_action: Optional[np.ndarray] = None
        self._latest_state: Optional[np.ndarray] = None
        self._state_joint_names: Optional[List[str]] = None
        self._initialized = False
        self._lock = threading.Lock()

        if not ROS_AVAILABLE:
            raise ImportError("rclpy is required for ROSArmStateListener")

        node.create_subscription(JointStateMsg, action_topic, self._action_cb, 10)
        node.create_subscription(JointStateMsg, state_topic, self._state_cb, 10)
        logging.info(f"[{label}] Listening: action={action_topic}  state={state_topic}")

    @staticmethod
    def _extract_positions(msg: Any) -> np.ndarray:
        return np.array(msg.position, dtype=np.float32)

    def _action_cb(self, msg: Any) -> None:
        positions = self._extract_positions(msg)
        if positions.size == 0:
            return
        with self._lock:
            self._latest_action = positions
        self._maybe_update()

    def _state_cb(self, msg: Any) -> None:
        positions = self._extract_positions(msg)
        with self._lock:
            self._latest_state = positions
            if self._state_joint_names is None:
                self._state_joint_names = list(msg.name) if hasattr(msg, "name") else []
        self._maybe_update()

    def _maybe_update(self) -> None:
        with self._lock:
            if (
                self._latest_action is None
                or self._latest_state is None
            ):
                return

            if not self._initialized:
                self._initialized = True
                logging.info(f"[{self._label}] All topics active — recording ready")

            action = self._latest_action.astype(np.float32)
            state = self._latest_state.copy()

        self._arm_state.update(action_qpos=action, state_qpos=state, synced=True)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._initialized

    def initial_state(self) -> Tuple[List[str], List[float]]:
        """Return (joint_names, positions) from the first state reading."""
        with self._lock:
            names = list(self._state_joint_names) if self._state_joint_names else []
            pos = self._latest_state.tolist() if self._latest_state is not None else []
            return names, pos


class ROSJointReader:
    """Subscribes to a joint_state topic and caches positions."""

    def __init__(self, node: "Node", topic: str, label: str = "") -> None:
        self._latest: Optional[np.ndarray] = None
        self._latest_names: List[str] = []
        self._lock = threading.Lock()
        self._ready = False

        if not ROS_AVAILABLE:
            raise ImportError("rclpy is required for ROSJointReader")

        node.create_subscription(JointStateMsg, topic, self._cb, 10)
        logging.info(f"[{label}] Subscribing to joint_state: {topic}")

    def _cb(self, msg: Any) -> None:
        with self._lock:
            self._latest = np.array(msg.position, dtype=np.float32)
            self._latest_names = list(msg.name) if hasattr(msg, "name") else []
            self._ready = True

    def get_joint_pos(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def get_joint_names(self) -> List[str]:
        with self._lock:
            return list(self._latest_names)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready


class RobotHomeSender:
    """Publishes a home joint state to each arm action topic.

    Optionally, after each arm reaches home, actively unrolls any wrap joints
    (e.g. ``wrist_3_joint``) that ended up wrapped by multiples of 2π. The
    unwrap is done by stepping the wrap joint(s) from their current readings
    toward the home target in ~``unwrap_max_step`` chunks (default π/2), one
    JointState waypoint per chunk. Non-wrap joints in each waypoint stay
    pinned at the home target so the arm doesn't drift while the wrist
    unrolls.
    """

    def __init__(
        self,
        node: "Node",
        arm_configs: dict[str, ArmTopicConfig],
        state_source: Optional[Callable[[str], Optional[np.ndarray]]] = None,
        wrap_joint_suffixes: Sequence[str] = (),
        unwrap_max_step: float = math.pi / 2,
        unwrap_waypoint_stamp_s: float = 1.0,
        unwrap_settle_tolerance: float = 0.05,
        unwrap_settle_timeout_s: Optional[float] = None,
    ) -> None:
        self._node = node
        self._arm_configs = arm_configs
        self._publishers = {
            name: node.create_publisher(JointStateMsg, cfg.action_topic, 10)
            for name, cfg in arm_configs.items()
        }
        self._state_source = state_source
        self._wrap_joint_suffixes = tuple(wrap_joint_suffixes or ())
        self._unwrap_max_step = float(unwrap_max_step)
        self._unwrap_waypoint_stamp_s = float(unwrap_waypoint_stamp_s)
        self._unwrap_settle_tolerance = float(unwrap_settle_tolerance)
        self._unwrap_settle_timeout_s = (
            float(unwrap_settle_timeout_s)
            if unwrap_settle_timeout_s is not None
            else max(2.0 * self._unwrap_waypoint_stamp_s, 2.0)
        )

    def send_home(
        self,
        home_positions: dict[str, List[float]],
        home_joint_names: dict[str, List[str]],
    ) -> None:
        if not ROS_AVAILABLE:
            raise ImportError("rclpy is required for RobotHomeSender")

        for arm_name, positions in home_positions.items():
            cfg = self._arm_configs.get(arm_name)
            if cfg is None:
                continue
            if arm_name not in self._publishers:
                continue
            joint_names = list(home_joint_names.get(arm_name, []))
            if not joint_names:
                logging.warning(f"[{arm_name}] No home joint names available; skipping home publish")
                continue
            msg = JointStateMsg()
            msg.header.stamp.sec = 10
            msg.header.stamp.nanosec = 0
            msg.name = joint_names
            msg.position = list(positions)
            self._publishers[arm_name].publish(msg)
            logging.info(f"[{arm_name}] Home pose published to {cfg.action_topic}")
            time.sleep(10)

            self._unwrap_wrap_joints(arm_name, joint_names, list(positions))

        time.sleep(5)

    def _unwrap_wrap_joints(
        self,
        arm_name: str,
        joint_names: List[str],
        home_positions: List[float],
    ) -> None:
        if not self._wrap_joint_suffixes or self._state_source is None:
            return

        wrap_indices = find_wrap_indices(joint_names, self._wrap_joint_suffixes)
        if not wrap_indices:
            return

        current = self._state_source(arm_name)
        if current is None:
            logging.warning(f"[{arm_name}] No current state available; skipping wrap-joint unwrap")
            return

        current_arr = np.asarray(current, dtype=np.float64)
        target_arr = np.asarray(home_positions, dtype=np.float64)
        min_len = min(current_arr.shape[0], target_arr.shape[0])
        valid_indices = [i for i in wrap_indices if i < min_len]
        if not valid_indices:
            logging.warning(
                f"[{arm_name}] Wrap-joint indices out of range for current/target vectors; "
                "skipping unwrap"
            )
            return

        deltas = {i: float(target_arr[i] - current_arr[i]) for i in valid_indices}
        max_abs_delta = max(abs(d) for d in deltas.values())
        if max_abs_delta <= self._unwrap_max_step:
            return

        num_steps = int(math.ceil(max_abs_delta / self._unwrap_max_step))
        logging.info(
            "[%s] Unwrapping wrap joints over %d waypoint(s): deltas=%s (indices=%s)",
            arm_name,
            num_steps,
            {joint_names[i]: round(d, 4) for i, d in deltas.items()},
            valid_indices,
        )

        for step_idx in range(1, num_steps + 1):
            alpha = step_idx / num_steps
            waypoint = list(target_arr)
            for i in valid_indices:
                waypoint[i] = float(current_arr[i] + alpha * deltas[i])
            msg = JointStateMsg()
            msg.header.stamp.sec = int(self._unwrap_waypoint_stamp_s)
            msg.header.stamp.nanosec = int(
                (self._unwrap_waypoint_stamp_s - int(self._unwrap_waypoint_stamp_s)) * 1e9
            )
            msg.name = list(joint_names)
            msg.position = [float(v) for v in waypoint]
            self._publishers[arm_name].publish(msg)
            # Block until the wrap joints actually reach this waypoint before
            # publishing the next one. If we race ahead (old behaviour: plain
            # time.sleep), the trajectory controller sees a fresh absolute
            # target while still mid-motion and can interpolate the "short
            # way round", re-wrapping the wrist we just tried to unroll.
            self._wait_for_waypoint_settled(
                arm_name=arm_name,
                wrap_indices=valid_indices,
                waypoint=np.asarray(waypoint, dtype=np.float64),
                step_idx=step_idx,
                num_steps=num_steps,
                joint_names=joint_names,
            )

    def _wait_for_waypoint_settled(
        self,
        arm_name: str,
        wrap_indices: List[int],
        waypoint: np.ndarray,
        step_idx: int,
        num_steps: int,
        joint_names: List[str],
    ) -> None:
        """Poll ``state_source`` until the wrap joints are within tolerance of ``waypoint``.

        Falls back to a fixed ``unwrap_waypoint_stamp_s`` sleep if no state
        source is wired up. Times out after ``unwrap_settle_timeout_s`` and
        logs a warning so a stuck arm can't deadlock the homing sequence.
        """
        nominal = max(self._unwrap_waypoint_stamp_s, 1e-3)
        if self._state_source is None:
            time.sleep(nominal)
            return

        poll_interval = max(0.02, min(0.1, nominal * 0.1))
        deadline = time.monotonic() + self._unwrap_settle_timeout_s
        last_err: Optional[float] = None
        while time.monotonic() < deadline:
            current = self._state_source(arm_name)
            if current is not None:
                current_arr = np.asarray(current, dtype=np.float64)
                if wrap_indices and max(wrap_indices) < current_arr.shape[0]:
                    errors = np.abs(current_arr[wrap_indices] - waypoint[wrap_indices])
                    last_err = float(np.max(errors))
                    if last_err <= self._unwrap_settle_tolerance:
                        return
            time.sleep(poll_interval)

        logging.warning(
            "[%s] wrap-joint waypoint %d/%d did not settle within %.2fs "
            "(max_err=%s rad, tol=%.3f); proceeding — check %s tracking",
            arm_name,
            step_idx,
            num_steps,
            self._unwrap_settle_timeout_s,
            f"{last_err:.3f}" if last_err is not None else "unknown",
            self._unwrap_settle_tolerance,
            [joint_names[i] for i in wrap_indices],
        )

def load_experiment_home_config(experiment_path: Path) -> Optional[dict]:
    if not experiment_path.exists():
        return None
    with open(experiment_path, "r") as f:
        return yaml.safe_load(f)


def save_experiment_home_config(
    experiment_path: Path,
    arm_home_positions: dict[str, List[float]],
    arm_joint_names: dict[str, List[str]],
    wrap_joints: Optional[Sequence[str]] = None,
) -> Path:
    experiment: dict = {}
    for arm_name, positions in arm_home_positions.items():
        experiment[f"{arm_name}_arm"] = {
            "ros__parameters": {
                "robot_joint_names": arm_joint_names.get(arm_name, []),
                "home_positions": positions,
            }
        }
    if wrap_joints is not None:
        experiment["wrap_joints"] = list(wrap_joints)

    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(experiment_path, "w") as f:
        yaml.safe_dump(experiment, f, sort_keys=False)
    return experiment_path


def extract_wrap_joints(experiment_config: Optional[dict]) -> Optional[List[str]]:
    """Return the wrap_joints list from an experiment_config.yaml dict, if any."""
    if not experiment_config:
        return None
    raw = experiment_config.get("wrap_joints")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return [str(x) for x in raw]


def extract_home_positions(experiment_config: Optional[dict]) -> dict[str, List[float]]:
    if not experiment_config:
        return {}

    home_positions: dict[str, List[float]] = {}
    for arm_name in ("left", "right"):
        section = experiment_config.get(f"{arm_name}_arm") or {}
        params = section.get("ros__parameters") or {}
        positions = params.get("home_positions") or []
        home_positions[arm_name] = list(positions)
    return home_positions


def extract_home_joint_names(experiment_config: Optional[dict]) -> dict[str, List[str]]:
    if not experiment_config:
        return {}

    home_joint_names: dict[str, List[str]] = {}
    for arm_name in ("left", "right"):
        section = experiment_config.get(f"{arm_name}_arm") or {}
        params = section.get("ros__parameters") or {}
        joint_names = params.get("robot_joint_names") or []
        home_joint_names[arm_name] = list(joint_names)
    return home_joint_names


PEDAL_DEFAULT_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-if01"

PEDAL_KEY_MAP = {
    "KEY_A": "toggle",
    "KEY_B": "discard",
    "KEY_C": "reset",
}


def resolve_pedal_evdev_paths(device_path: str) -> List[str]:
    """Return all evdev nodes for a composite USB pedal (e.g. if00 + if01)."""
    p = Path(device_path)
    parent = p.parent
    name = p.name
    if not parent.is_dir() or "-event-" not in name:
        return [device_path]

    base = name.split("-event-", 1)[0]
    matches = sorted(parent.glob(f"{base}-event-*"))
    if matches:
        return [str(x.resolve()) for x in matches]
    return [device_path]


class FootPedalThread(threading.Thread):
    """Background thread that reads a USB foot pedal via evdev."""

    def __init__(
        self,
        device_path: str,
        stop_event: threading.Event,
        on_toggle: Callable[[], None],
        on_discard: Callable[[], None],
        on_reset: Callable[[], None],
    ) -> None:
        super().__init__(daemon=True, name="foot-pedal")
        self.device_path = device_path
        self._stop_event = stop_event
        self._on_toggle = on_toggle
        self._on_discard = on_discard
        self._on_reset = on_reset

    def run(self) -> None:
        paths = resolve_pedal_evdev_paths(self.device_path)
        devices: List[InputDevice] = []
        for path in paths:
            try:
                device = InputDevice(path)
                devices.append(device)
                logging.info("Foot pedal opened: %s (%s)", device.path, device.name)
            except PermissionError:
                logging.error(
                    "Permission denied for %s. Try:  sudo chmod a+r %s  or add a udev rule.",
                    path, path,
                )
            except FileNotFoundError:
                logging.error("Foot pedal device not found: %s", path)

        if not devices:
            return

        try:
            grabbed = 0
            for device in devices:
                try:
                    device.grab()
                    grabbed += 1
                except OSError as exc:
                    logging.warning(
                        "Could not grab %s exclusively (%s); that node may still type into the shell.",
                        device.path,
                        exc,
                    )
            if grabbed:
                logging.info(
                    "Foot pedal exclusive grab on %d/%d node(s) — pedal keys should not reach the terminal.",
                    grabbed,
                    len(devices),
                )

            fds = [device.fd for device in devices]
            fd_to_dev = {device.fd: device for device in devices}
            while not self._stop_event.is_set():
                try:
                    readable, _, _ = select.select(fds, [], [], 0.2)
                except (ValueError, OSError):
                    break
                if self._stop_event.is_set():
                    break
                for fd in readable:
                    device = fd_to_dev[fd]
                    try:
                        while True:
                            event = device.read_one()
                            if event is None:
                                break
                            if event.type != ev_ecodes.EV_KEY:
                                continue
                            key = categorize(event)
                            if key.keystate != 1:
                                continue
                            keycode = key.keycode if isinstance(key.keycode, str) else key.keycode[0]
                            self._dispatch(PEDAL_KEY_MAP.get(keycode))
                    except OSError:
                        if not self._stop_event.is_set():
                            logging.error("Foot pedal read error on %s", device.path)
                        break
        finally:
            for device in devices:
                try:
                    device.ungrab()
                except OSError:
                    pass

    def _dispatch(self, action: str | None) -> None:
        if action == "toggle":
            self._on_toggle()
        elif action == "discard":
            self._on_discard()
        elif action == "reset":
            self._on_reset()


# ── Shared record/deploy helpers ──────────────────────────────────────────


def resolve_unwrap_config(cfg: dict) -> Dict[str, Optional[float]]:
    """Parse the optional ``unwrap:`` section of gello.yaml.

    Returns a dict with ``max_step`` (radians per unwrap waypoint, default
    π/2), ``waypoint_stamp_s`` (trajectory deadline per waypoint, default
    1.0s), ``settle_tolerance`` (radians of wrap-joint error we consider
    "arrived" before publishing the next waypoint, default 0.05 ≈ 3°) and
    ``settle_timeout_s`` (hard cap before we warn and move on; ``None``
    means ``max(2 × waypoint_stamp_s, 2.0)``). Unknown keys are ignored;
    missing section yields defaults so ``RobotHomeSender`` can be
    constructed uniformly.
    """
    defaults = {
        "max_step": math.pi / 2,
        "waypoint_stamp_s": 1.0,
        "settle_tolerance": 0.05,
        "settle_timeout_s": None,
    }
    raw = cfg.get("unwrap") if isinstance(cfg, dict) else None
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError(
            "`unwrap:` must be a mapping with max_step / waypoint_stamp_s / "
            "settle_tolerance / settle_timeout_s"
        )
    out = dict(defaults)
    if "max_step" in raw:
        out["max_step"] = float(raw["max_step"])
    if "waypoint_stamp_s" in raw:
        out["waypoint_stamp_s"] = float(raw["waypoint_stamp_s"])
    if "settle_tolerance" in raw:
        out["settle_tolerance"] = float(raw["settle_tolerance"])
    if "settle_timeout_s" in raw and raw["settle_timeout_s"] is not None:
        out["settle_timeout_s"] = float(raw["settle_timeout_s"])
    return out


class WrapJointManager:
    """Tracks wrap-joint indices and per-episode 2π offsets for any number of arms.

    Recording-side usage:
        mgr.configure(arm_joint_names)
        # per frame:
        state, action = mgr.subtract_state_action(arm, state, action, episode_idx)
        # at episode start:
        mgr.reset_episode()

    Deploy-side usage:
        mgr.configure(arm_joint_names)
        state = mgr.subtract_state(arm, state, episode_idx)
        action = mgr.add_action(arm, action)
    """

    _ACTION_BAND = 2.0 * math.pi

    def __init__(self, wrap_joint_suffixes: Sequence[str]) -> None:
        self.wrap_joint_suffixes: List[str] = list(wrap_joint_suffixes or [])
        self._indices: Dict[str, List[int]] = {}
        self._joint_names: Dict[str, List[str]] = {}
        self._offsets: Dict[str, Optional[np.ndarray]] = {}
        self._out_of_band_logged: Dict[str, bool] = {}

    def configure(self, arm_joint_names: Dict[str, List[str]]) -> None:
        """Compute which joints match the suffix list per arm and log the result."""
        self._indices = {}
        self._joint_names = {arm: list(names) for arm, names in arm_joint_names.items()}
        for arm, names in self._joint_names.items():
            self._indices[arm] = find_wrap_indices(names, self.wrap_joint_suffixes)

        if not self.wrap_joint_suffixes:
            logging.info("wrap_joints disabled (empty list); recording raw joint angles as-is")
            return

        for arm, indices in self._indices.items():
            names = self._joint_names[arm]
            if indices:
                joint_hits = [names[i] for i in indices]
                logging.info(
                    "[%s] wrap_joints active: %s (indices %s) — per-episode offset will be subtracted from state/action",
                    arm,
                    joint_hits,
                    indices,
                )
            else:
                logging.info(
                    "[%s] wrap_joints configured (%s) but no matching joints in %s",
                    arm,
                    self.wrap_joint_suffixes,
                    names,
                )

    def indices(self, arm: str) -> List[int]:
        return list(self._indices.get(arm, []))

    def has_wrap(self, arm: str) -> bool:
        return bool(self._indices.get(arm))

    def offset(self, arm: str) -> Optional[np.ndarray]:
        return self._offsets.get(arm)

    def reset_episode(self, arm: Optional[str] = None) -> None:
        """Clear the latched 2π offset so the next frame re-latches."""
        if arm is None:
            self._offsets = {}
            self._out_of_band_logged = {}
        else:
            self._offsets.pop(arm, None)
            self._out_of_band_logged.pop(arm, None)

    def prelatch_offset(
        self,
        arm: str,
        reference_positions: Sequence[float] | np.ndarray,
        episode_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Seed the per-episode 2π offset from a known reference vector.

        Typical use: immediately after homing (``RobotHomeSender.send_home``),
        pass in ``home_positions[arm]`` so the offset is derived from the
        pose the controller was explicitly commanded to reach, rather than
        from the first live state sample. Without this, a single stale
        pre-homing state read racing the ROS subscriber callback can cause
        ``_latch_offset`` to latch an offset of ±2π, and the very next
        ``add_action`` will then re-wrap the joint back to its pre-home
        position — the "home → unwrap → jump back to wrap" symptom.

        Later ``subtract_state`` / ``subtract_state_action`` calls see the
        offset is already set and reuse it verbatim, so this is safe to
        call even when the arm has no wrap joints (no-op).
        """
        indices = self._indices.get(arm) or []
        if not indices:
            return None
        ref_arr = np.asarray(reference_positions, dtype=np.float64)
        if ref_arr.size == 0 or max(indices) >= ref_arr.size:
            return None
        q0 = ref_arr[indices]
        latched = (q0 - wrap_pi(q0)).astype(np.float32)
        self._offsets[arm] = latched
        self._out_of_band_logged.pop(arm, None)
        if np.any(np.abs(latched) > 1e-6):
            logging.info(
                "[%s] wrap_joints offset pre-latched from home for episode %d: %s rad (indices %s)",
                arm,
                episode_idx,
                latched.tolist(),
                indices,
            )
        return latched

    def _latch_offset(self, arm: str, state: np.ndarray, episode_idx: int) -> Optional[np.ndarray]:
        indices = self._indices.get(arm) or []
        if not indices:
            return None
        offset = self._offsets.get(arm)
        if offset is not None:
            return offset
        q0 = state[indices].astype(np.float64)
        latched = (q0 - wrap_pi(q0)).astype(state.dtype)
        self._offsets[arm] = latched
        if np.any(np.abs(latched) > 1e-6):
            logging.info(
                "[%s] wrap_joints offset latched for episode %d: %s rad (indices %s)",
                arm,
                episode_idx,
                latched.tolist(),
                indices,
            )
        return latched

    def _maybe_log_out_of_band(self, arm: str, action_slice: np.ndarray) -> None:
        if self._out_of_band_logged.get(arm):
            return
        if np.any(np.abs(action_slice) > self._ACTION_BAND):
            logging.warning(
                "[%s] wrap_joint action %s exceeds ±2π after per-episode offset — "
                "the task is rotating past one full turn from the episode start, "
                "which will push the policy off its training distribution at deploy. "
                "Consider breaking the task into shorter episodes.",
                arm,
                action_slice.tolist(),
            )
            self._out_of_band_logged[arm] = True

    def subtract_state_action(
        self,
        arm: str,
        state: np.ndarray,
        action: np.ndarray,
        episode_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Record-side: latch offset from state, subtract it from state and action."""
        indices = self._indices.get(arm) or []
        if not indices:
            return state, action
        state = np.array(state, copy=True)
        action = np.array(action, copy=True)
        offset = self._latch_offset(arm, state, episode_idx)
        if offset is None:
            return state, action
        state[indices] = state[indices] - offset
        action[indices] = action[indices] - offset
        self._maybe_log_out_of_band(arm, action[indices])
        return state, action

    def subtract_state(
        self,
        arm: str,
        state: np.ndarray,
        episode_idx: int = 0,
    ) -> np.ndarray:
        """Deploy-side: latch offset from state, subtract it from state only."""
        indices = self._indices.get(arm) or []
        if not indices:
            return state
        state = np.array(state, copy=True)
        offset = self._latch_offset(arm, state, episode_idx)
        if offset is None:
            return state
        state[indices] = state[indices] - offset
        return state

    def add_action(self, arm: str, action: np.ndarray) -> np.ndarray:
        """Deploy-side: add latched offset back onto action before publishing."""
        indices = self._indices.get(arm) or []
        if not indices:
            return action
        offset = self._offsets.get(arm)
        if offset is None:
            return action
        action = np.array(action, copy=True)
        action[indices] = action[indices] + offset.astype(action.dtype)
        return action


class CameraStabilizer:
    """Per-camera 180° orientation stabilization with a latched previous frame.

    Wraps the boilerplate of ``stabilize_flags`` + ``prev_frames`` used in both
    the recording loop and the deploy loop. ``process(frames)`` walks the list,
    applying ``orientation_stabilize_180_vs_prev`` only where the flag is on
    and returning a new list with the stabilized frames in place.
    """

    def __init__(
        self,
        stabilize_flags: Sequence[bool],
        mae_thresholds: Sequence[float],
    ) -> None:
        self.stabilize_flags: List[bool] = [bool(f) for f in stabilize_flags]
        self.mae_thresholds: List[float] = [float(t) for t in mae_thresholds]
        if len(self.mae_thresholds) != len(self.stabilize_flags):
            raise ValueError("stabilize_flags and mae_thresholds must be the same length")
        self._prev: List[Optional[np.ndarray]] = [None] * len(self.stabilize_flags)

    @property
    def num_cameras(self) -> int:
        return len(self.stabilize_flags)

    @property
    def active_count(self) -> int:
        return sum(1 for f in self.stabilize_flags if f)

    def reset(self) -> None:
        self._prev = [None] * len(self.stabilize_flags)

    def process(self, frames: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        out: List[Optional[np.ndarray]] = []
        for i, frame in enumerate(frames):
            if i >= len(self.stabilize_flags) or not self.stabilize_flags[i] or frame is None:
                out.append(frame)
                continue
            stabilized, self._prev[i] = orientation_stabilize_180_vs_prev(
                self._prev[i], frame, self.mae_thresholds[i]
            )
            out.append(stabilized)
        return out


def resolve_home(
    experiment_config: Optional[dict],
    arm_keys: Sequence[str],
    state_getter: Callable[[str], Tuple[List[str], List[float]]],
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    """Return ``(home_positions, home_joint_names)`` for the given arms.

    Values are taken from ``experiment_config`` first (as saved by record.py),
    falling back to ``state_getter(arm)`` — a callable that returns
    ``(joint_names, positions)`` — for anything missing. This consolidates the
    ``build_home_snapshot`` / ``resolve_joint_names`` dance that both scripts
    used to do inline.
    """
    positions = extract_home_positions(experiment_config)
    names = extract_home_joint_names(experiment_config)

    out_positions: Dict[str, List[float]] = {}
    out_names: Dict[str, List[str]] = {}
    for arm in arm_keys:
        arm_names = list(names.get(arm) or [])
        arm_positions = list(positions.get(arm) or [])
        if not arm_names or not arm_positions:
            fetched_names, fetched_positions = state_getter(arm)
            if not arm_names:
                arm_names = list(fetched_names or [])
            if not arm_positions:
                arm_positions = list(fetched_positions or [])
        out_names[arm] = arm_names
        out_positions[arm] = arm_positions
    return out_positions, out_names


def build_recording_snapshot(
    base_cfg: dict,
    camera_configs: Sequence[CameraConfig],
    readers_by_name: Dict[str, "CameraReader"],
) -> dict:
    """Return a copy of ``base_cfg`` with the ``cameras:`` tree expanded.

    The expanded tree records, per camera, the port, resolution, orientation
    flags, ``requested_settings`` (operator intent), and ``applied_settings``
    (v4l2 read-back). It replaces the previous split between
    ``recording_config.yaml`` and ``camera_settings_applied.yaml`` so that
    deploy.py has a single file to reparse.
    """
    resolved = dict(base_cfg)
    resolved["cameras"] = {
        camera.name: {
            "port": camera.port,
            **({"resolution": list(camera.resolution)} if camera.resolution else {}),
            "stabilize_orientation_180": camera.stabilize_orientation_180,
            "orientation_mae_delta_threshold": camera.orientation_mae_delta_threshold,
            **(
                {"requested_settings": dict(camera.settings)}
                if camera.settings else {}
            ),
            **(
                {"applied_settings": dict(readers_by_name[camera.name].applied_settings)}
                if camera.name in readers_by_name and readers_by_name[camera.name].applied_settings
                else {}
            ),
        }
        for camera in camera_configs
    }
    for section_name in ("left_cameras", "right_cameras", "other_cameras"):
        resolved.pop(section_name, None)
    return resolved


def read_camera_overrides(cfg: dict, recording_config_path: Path) -> dict:
    """Overwrite ``cfg``'s camera_defaults/per-camera settings from a recording_config.yaml.

    The goal is to reproduce, at deploy time, exactly the v4l2 state the
    cameras were in when the training dataset was recorded:

    - ``camera_defaults.settings`` is replaced wholesale with whatever the
      recording snapshot has under the same key.
    - Each camera's ``settings:`` block becomes the subset of
      ``requested_settings`` (or ``applied_settings`` as fallback) that
      differs from the new defaults, so the existing merge logic in
      ``load_camera_configs`` reproduces the full applied set.

    Cameras present in ``cfg`` but missing from the snapshot keep their
    existing overrides; cameras present in the snapshot but not in ``cfg``
    are skipped. Ports/resolutions are untouched — those describe where the
    cameras live on the USB bus today, not what the training rig looked like.
    """
    with open(recording_config_path, "r") as fh:
        snapshot = yaml.safe_load(fh) or {}

    snap_defaults = ((snapshot.get("camera_defaults") or {}).get("settings")) or {}
    snap_defaults = {str(k): int(v) for k, v in snap_defaults.items()}
    cfg.setdefault("camera_defaults", {})["settings"] = dict(snap_defaults)

    snap_cameras = snapshot.get("cameras") or {}

    def _pick(settings_block: Optional[dict]) -> Dict[str, int]:
        if not isinstance(settings_block, dict):
            return {}
        return {str(k): int(v) for k, v in settings_block.items()}

    matched: List[str] = []
    unmatched_in_cfg: List[str] = []
    unmatched_in_snap = set(snap_cameras.keys())

    def _apply_to_section(section: Any) -> None:
        if not isinstance(section, dict):
            return
        for camera_name, entry in section.items():
            if not isinstance(entry, dict):
                continue
            snap_entry = snap_cameras.get(camera_name)
            if not isinstance(snap_entry, dict):
                unmatched_in_cfg.append(camera_name)
                continue
            unmatched_in_snap.discard(camera_name)
            snap_full = _pick(snap_entry.get("requested_settings")) or _pick(
                snap_entry.get("applied_settings")
            )
            overrides = {k: v for k, v in snap_full.items() if snap_defaults.get(k) != v}
            entry["settings"] = overrides
            matched.append(camera_name)

    for section_name in ("left_cameras", "right_cameras", "other_cameras", "cameras"):
        _apply_to_section(cfg.get(section_name))

    logging.info(
        "Camera settings overridden from %s (matched=%s, no-snapshot-entry=%s, extra-in-snapshot=%s)",
        recording_config_path,
        sorted(matched),
        sorted(unmatched_in_cfg),
        sorted(unmatched_in_snap),
    )
    return cfg


def run_cbreak_keyboard_loop(
    stop_event: threading.Event,
    handlers: Dict[str, Callable[[], None]],
    quit_keys: Sequence[str] = ("q", "\x1b"),
    poll_interval_s: float = 0.1,
) -> None:
    """Read single characters from stdin in cbreak mode and dispatch them.

    - ``handlers`` maps a lowercase character to a callable.
    - Any key in ``quit_keys`` sets ``stop_event`` and exits the loop.
    - If stdin is not a TTY, logs a warning and returns immediately (so the
      caller can still rely on external controllers, e.g. a foot pedal).
    """
    import termios
    import tty

    if not sys.stdin.isatty():
        logging.warning(
            "stdin is not a TTY; keyboard control is unavailable in this mode"
        )
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], poll_interval_s)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if not key:
                continue
            if key in quit_keys:
                stop_event.set()
                break
            handler = handlers.get(key.lower())
            if handler is not None:
                handler()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    print("Oooops, this is just a helper module, not meant to be run directly.")