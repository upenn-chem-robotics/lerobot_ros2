#!/usr/bin/env python3
"""
Deploy a trained policy (ACT / Diffusion) on UR3e bimanual arms via ROS 2.

Pure ROS 2 pub/sub: subscribes to UR joint states for observations,
runs policy inference, and publishes joint commands back. The
joint_state_to_trajectory_node (from ur_robotiq) converts these to
trajectory commands for the UR controllers.

Controls (keyboard via OpenCV window):
  space — start / e-stop toggle
  q     — quit

Usage:
    # Terminal 1: launch ur_robotiq WITHOUT GELLO
    ros2 launch ur_robotiq control_bimanual_ur3_robotiq.launch.py use_gello:=false ...

    # Terminal 2: deploy policy
    lerobot-ros-deploy --policy outputs/act_pick_place/checkpoints/last/pretrained_model
    lerobot-ros-deploy --policy outputs/dp_pick_place/checkpoints/last/pretrained_model --visualize
    XXX
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import torch
import yaml
from datetime import datetime
from torchvision.transforms import v2 as transforms_v2

os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)

from lerobot_ros2.config_paths import resolve_config_path
from lerobot_ros2.helper import (
    CameraStabilizer,
    ROSJointReader,
    RobotHomeSender,
    WrapJointManager,
    extract_home_joint_names,
    load_arm_configs,
    load_camera_configs,
    load_config,
    load_experiment_home_config,
    open_configured_cameras,
    read_camera_overrides,
    resolve_home,
    resolve_unwrap_config,
    resolve_wrap_joints,
    run_cbreak_keyboard_loop,
)
from lerobot_ros2.visualizer import LivePreview

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from lerobot_ros2.strided_history import StridedHistoryRunner, load_strided_config


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


# ── Auto-detect training resize ──────────────────────────────────────────

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


def resolve_experiment_config_path(policy_path: str, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    base_path = Path(policy_path)
    if not base_path.exists():
        return None

    for parent in (base_path, *base_path.parents):
        for candidate_name in ("experiment_config.yaml", "recording_config.yaml", "config.yaml"):
            candidate = parent / candidate_name
            if candidate.exists():
                return candidate
    return None


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


def _policy_path_exists(policy_path: str) -> bool:
    return Path(policy_path).exists()


def _load_policy_type(policy_path: str, explicit_mode: str) -> str:
    if explicit_mode != "auto":
        return "diffusion" if explicit_mode == "dp" else "act"

    if not _policy_path_exists(policy_path):
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
    return raw_type


def _image_feature_keys(policy) -> List[str]:
    return [
        name
        for name, feature in policy.config.input_features.items()
        if feature.type == "VISUAL" and name.startswith("observation.images.")
    ]


def _policy_visual_size(policy) -> Optional[Tuple[int, int]]:
    for feature in policy.config.input_features.values():
        if feature.type == "VISUAL" and len(feature.shape) == 3:
            return int(feature.shape[1]), int(feature.shape[2])
    return None


def _resolve_arm_keys(
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


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy trained policy via ROS 2.")
    parser.add_argument("--policy", required=True,
                        help="Path to pretrained model dir or HuggingFace repo id.")
    parser.add_argument(
        "--deploy-mode",
        type=str,
        default="auto",
        choices=("auto", "act", "dp"),
        help="Force a deploy path. Use dp for DiffusionPolicy checkpoints.",
    )
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control frequency in Hz (default: 10).")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds while running.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the named camera / teleop config file. "
                             "Defaults to $GELLO_CONFIG.")
    parser.add_argument("--experiment-config", type=str, default=None,
                        help="Path to experiment_config.yaml saved with the dataset.")
    parser.add_argument(
        "--camera-settings",
        type=str,
        default=None,
        help=(
            "Path to the recording_config.yaml saved with the training "
            "dataset. Overrides the camera_defaults / per-camera settings "
            "blocks from gello.yaml so the cameras match the state they were "
            "in at recording time. If omitted, deploy.py auto-discovers a "
            "recording_config.yaml in the policy directory or its parents; "
            "if none is found, gello.yaml values are used as-is. "
            "Pass --camera-settings '' to force using gello.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not ROS_AVAILABLE:
        sys.exit("rclpy not found. Source your ROS 2 workspace first:\n  source /opt/ros/humble/setup.bash")

    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)

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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA not available, falling back to CPU")

    # ── Load policy ──────────────────────────────────────────────────────
    logging.info(f"Loading policy from {args.policy}...")
    policy_type = _load_policy_type(args.policy, args.deploy_mode)
    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(args.policy)
    else:
        policy = ACTPolicy.from_pretrained(args.policy)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    if policy_type == "diffusion" and hasattr(policy, "reset"):
        policy.reset()

    # If the checkpoint was trained with a uniform-stride observation
    # window (policy.type == "strided_diffusion"; see the
    # lerobot_policy_strided_diffusion plugin), swap in a runner that keeps
    # its own strided ring buffer and drives predict_action_chunk directly.
    # The policy itself stays untouched.
    strided_cfg = None
    strided_runner: Optional[StridedHistoryRunner] = None
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
                    "Strided history expects %d Hz inference (matches training fps); "
                    "you requested --hz %.2f. The 1-second stride will be off unless "
                    "--hz matches the dataset fps.",
                    expected_hz,
                    args.hz,
                )
            logging.info(
                "Strided history enabled: n_obs_steps=%d, stride=%.2fs, "
                "buffer_length=%d frames",
                strided_runner.n_obs,
                strided_runner.stride_seconds,
                strided_runner.buffer_length,
            )
    logging.info(f"Policy loaded: {policy.config.type} on {device}")
    logging.info(f"  Input:  {list(policy.config.input_features.keys())}")
    logging.info(f"  Output: {list(policy.config.output_features.keys())}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
    )

    image_feature_keys = _image_feature_keys(policy)
    expected_state_dim = policy.config.robot_state_feature.shape[0] if policy.config.robot_state_feature is not None else None
    policy_visual_size = _policy_visual_size(policy)
    if policy_type == "diffusion":
        logging.info(f"DP observation keys: {image_feature_keys + ['observation.state']}")
        if expected_state_dim is not None:
            logging.info(f"Expected state dimension: {expected_state_dim}")

    training_resize = detect_training_resize(args.policy)
    resize_transform = None
    if training_resize:
        resize_transform = transforms_v2.Resize(list(training_resize), antialias=True)
        logging.info(f"Auto-detected training resize: {training_resize[0]}x{training_resize[1]}")
    elif policy_visual_size:
        resize_transform = transforms_v2.Resize(list(policy_visual_size), antialias=True)
        logging.info(
            "No training resize found; falling back to policy visual size: %sx%s",
            policy_visual_size[0],
            policy_visual_size[1],
        )

    # ── Cameras ──────────────────────────────────────────────────────────
    camera_configs = load_camera_configs(cfg)
    logging.info("Opening configured cameras: %s", [f"{camera.index}:{camera.name}" for camera in camera_configs])
    cameras = open_configured_cameras(camera_configs)
    time.sleep(0.3)
    logging.info(f"{len(cameras)} camera(s) active")

    readers_by_name = {reader.name: reader for reader in cameras}

    # ── 180° orientation stabilization (mirrors record.py) ──────────────
    # Some cameras occasionally deliver an upside-down frame after a USB hiccup.
    # The CameraStabilizer compares each frame against the previous stabilized
    # frame and swaps in the 180°-rotated version when it matches better by at
    # least ``orientation_mae_delta_threshold``. The buffer resets at the start
    # of every episode via ``_start_new_episode``.
    configs_by_name = {c.name: c for c in camera_configs}
    stabilize_flags: List[bool] = [
        bool(configs_by_name[c.name].stabilize_orientation_180) if c.name in configs_by_name else False
        for c in cameras
    ]
    orientation_mae_thresholds: List[float] = [
        float(configs_by_name[c.name].orientation_mae_delta_threshold) if c.name in configs_by_name else 20.0
        for c in cameras
    ]
    stabilizer = CameraStabilizer(stabilize_flags, orientation_mae_thresholds)
    for cam, flag, thr in zip(cameras, stabilize_flags, orientation_mae_thresholds):
        if flag:
            logging.info(
                "[%s] orientation-180 stabilization active (mae_delta_threshold=%.1f)",
                cam.name or f"cam{cam.index}",
                thr,
            )

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_dir = Path(args.policy) if Path(args.policy).is_dir() else Path(args.policy).parent
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-run deployment video recording ───────────────────────────────
    deploy_run_dir = snapshot_dir / "deployments" / run_stamp
    video_writers: Dict[str, cv2.VideoWriter] = {}
    video_paths: Dict[str, Path] = {}
    frame_counts: Dict[str, int] = {}
    try:
        deploy_run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.warning("Could not create deploy recording dir %s: %s", deploy_run_dir, exc)
        deploy_run_dir = None  # type: ignore[assignment]

    if deploy_run_dir is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam in cameras:
            video_path = deploy_run_dir / f"{cam.name}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                float(args.hz),
                (int(cam.width), int(cam.height)),
            )
            if not writer.isOpened():
                logging.warning(
                    "Could not open VideoWriter for %s at %s (size=%sx%s, fps=%s); "
                    "skipping recording for this camera.",
                    cam.name,
                    video_path,
                    cam.width,
                    cam.height,
                    args.hz,
                )
                try:
                    writer.release()
                except Exception:
                    pass
                continue
            video_writers[cam.name] = writer
            video_paths[cam.name] = video_path
            frame_counts[cam.name] = 0
            logging.info(
                "Recording %s -> %s (%sx%s @ %s Hz)",
                cam.name,
                video_path,
                cam.width,
                cam.height,
                args.hz,
            )

        # Manifest for this run, co-located with the videos. This is the sole
        # record of what v4l2 settings actually stuck at deploy time (the
        # record-side equivalent lives in recording_config.yaml).
        manifest = {
            "run_stamp": run_stamp,
            "deployed_at": datetime.now().isoformat(timespec="seconds"),
            "policy_path": str(args.policy),
            "hz": float(args.hz),
            "camera_settings_source": str(camera_settings_path) if camera_settings_path else str(config_path),
            "camera_defaults": cfg.get("camera_defaults") or {},
            "cameras": {
                cam.name: {
                    "port": cam.device,
                    "video_file": video_paths[cam.name].name if cam.name in video_paths else None,
                    "width": int(cam.width),
                    "height": int(cam.height),
                    "requested_settings": dict(
                        configs_by_name[cam.name].settings
                    ) if cam.name in configs_by_name else {},
                    "applied_settings": dict(
                        readers_by_name[cam.name].applied_settings
                    ) if cam.name in readers_by_name else {},
                }
                for cam in cameras
            },
        }
        manifest_path = deploy_run_dir / "deployment_info.yaml"
        try:
            with open(manifest_path, "w") as fh:
                yaml.safe_dump(manifest, fh, sort_keys=False)
            logging.info("Wrote deployment manifest to %s", manifest_path)
        except OSError as exc:
            logging.warning("Could not write deployment manifest to %s: %s", manifest_path, exc)
    if len(cameras) != len(image_feature_keys):
        logging.error(
            "Number of cameras (%s) does not match policy visual inputs (%s). "
            "Policy expects these keys in camera config order: %s. Adjust %s or the checkpoint.",
            len(cameras),
            len(image_feature_keys),
            image_feature_keys,
            config_path,
        )
        sys.exit(1)
    logging.info("Mapping cameras to policy image keys (same order): %s", image_feature_keys)

    # ── ROS 2 node ───────────────────────────────────────────────────────
    rclpy.init()
    node = rclpy.create_node("policy_deployer")

    arm_configs = load_arm_configs(cfg)

    experiment_config_path = resolve_experiment_config_path(args.policy, args.experiment_config)
    experiment_config = load_experiment_home_config(experiment_config_path) if experiment_config_path else None
    if experiment_config_path:
        logging.info(f"Using experiment config: {experiment_config_path}")

    # Pick the active arms based on which entries have joint names in the
    # experiment config vs. what the policy expects.
    arm_joint_names_from_cfg = extract_home_joint_names(experiment_config)
    arm_keys = _resolve_arm_keys(arm_configs, arm_joint_names_from_cfg, policy)
    logging.info(f"Active arm(s): {arm_keys}")

    # Readers for arm state
    state_readers: Dict[str, ROSJointReader] = {}
    for key in arm_keys:
        arm_cfg = arm_configs[key]
        state_readers[key] = ROSJointReader(node, arm_cfg.state_topic, label=key)

    # Spin ROS in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── Wait for UR state ────────────────────────────────────────────────
    logging.info("Waiting for UR joint state topics...")
    while True:
        if all(r.is_ready for r in state_readers.values()):
            break
        time.sleep(0.1)
    logging.info("All UR state topics active")

    def _reader_state(arm_name: str) -> Tuple[List[str], List[float]]:
        reader = state_readers.get(arm_name)
        if reader is None:
            return [], []
        qpos = reader.get_joint_pos()
        return reader.get_joint_names(), qpos.tolist() if qpos is not None else []

    home_positions, arm_joint_names = resolve_home(experiment_config, arm_keys, _reader_state)
    for key in arm_keys:
        if not arm_joint_names.get(key):
            logging.error(
                f"Could not determine joint names for {key} arm. Make sure the experiment config "
                "has 'home' joint names or that the ROS topic is publishing."
            )
            sys.exit(1)

    try:
        wrap_joint_suffixes = resolve_wrap_joints(cfg)
    except ValueError as exc:
        sys.exit(f"Invalid wrap_joints in recording config: {exc}")
    unwrap_opts = resolve_unwrap_config(cfg)

    home_sender = RobotHomeSender(
        node,
        arm_configs,
        state_source=lambda arm: state_readers[arm].get_joint_pos() if arm in state_readers else None,
        wrap_joint_suffixes=wrap_joint_suffixes,
        unwrap_max_step=unwrap_opts["max_step"],
        unwrap_waypoint_stamp_s=unwrap_opts["waypoint_stamp_s"],
        unwrap_settle_tolerance=unwrap_opts["settle_tolerance"],
        unwrap_settle_timeout_s=unwrap_opts["settle_timeout_s"],
    )

    # Publishers for arm actions
    cmd_publishers: Dict[str, ROSActionPublisher] = {}
    for key in arm_keys:
        arm_cfg = arm_configs[key]
        cmd_publishers[key] = ROSActionPublisher(
            node,
            arm_cfg.action_topic,
            arm_joint_names[key],
            label=key,
        )
    expected_action_dim = sum(len(names) for names in arm_joint_names.values())
    if policy_type == "diffusion" and expected_state_dim is not None and expected_state_dim != expected_action_dim:
        logging.warning(
            "Policy expects a %s-D state/action, but the configured arms expose %s joints.",
            expected_state_dim,
            expected_action_dim,
        )

    # ── Wrap-joints (per-episode revolution offset, mirrors record.py) ───
    # At the start of every episode we latch ``offset = q0 - wrap_pi(q0)``
    # (nearest multiple of 2π) from the first state reading. The offset is
    # subtracted from state before the policy sees it (policy receives the
    # wrapped range it was trained on) and added back onto the commanded
    # action before publishing (so the robot controller gets the absolute
    # angle). All of this is owned by WrapJointManager.
    wrap_manager = WrapJointManager(wrap_joint_suffixes)
    wrap_manager.configure({key: arm_joint_names[key] for key in arm_keys})

    # ── Events ───────────────────────────────────────────────────────────
    stop_event = threading.Event()
    running_event = threading.Event()
    _force_count = 0

    def _shutdown(sig, frame):
        nonlocal _force_count
        _force_count += 1
        if _force_count >= 2:
            logging.warning("Force exit")
            os._exit(1)
        logging.info("Shutting down (Ctrl-C again to force)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Control loop ─────────────────────────────────────────────────────
    period = 1.0 / args.hz
    step_count = 0
    episode_count = 0

    logging.info(
        f"\nReady — {len(arm_keys)} arm(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        f"  Press SPACE -> replay policy from home\n"
        f"  Press Q     -> quit\n"
        f"  Waiting for SPACE to replay...\n"
    )

    preview: Optional[LivePreview] = None
    cam_labels: List[str] = []
    if cameras and args.visualize:
        cam_labels = [c.display_label if getattr(c, "display_label", "") else f"cam {c.index}" for c in cameras]

    terminal_keyboard_enabled = not args.visualize

    def _start_new_episode() -> None:
        nonlocal episode_count
        running_event.clear()
        if strided_runner is not None:
            strided_runner.reset()
        elif hasattr(policy, "reset"):
            policy.reset()
        if hasattr(preprocessor, "reset"):
            preprocessor.reset()
        if hasattr(postprocessor, "reset"):
            postprocessor.reset()
        wrap_manager.reset_episode()
        stabilizer.reset()
        logging.info("REPLAY — homing, then restarting policy")
        home_sender.send_home(home_positions, arm_joint_names)
        episode_count += 1
        # Seed wrap-joint offsets from the home pose we just commanded, not
        # from a live state snapshot. The ROS state subscriber runs in a
        # background thread and the most recent cached sample at this point
        # can race the post-homing update; if it's still the pre-homing
        # wrap (e.g. wrist_3 at ±2π), `_latch_offset` would pick up that
        # value and the very first `add_action` would send the arm back
        # to the wrapped pose. Pre-latching from `home_positions` removes
        # the race entirely.
        for arm_name in arm_keys:
            home_arr = home_positions.get(arm_name) or []
            if home_arr:
                wrap_manager.prelatch_offset(arm_name, home_arr, episode_count)
        running_event.set()
        logging.info(f"REPLAY — policy running (episode {episode_count})")

    if terminal_keyboard_enabled:
        threading.Thread(
            target=run_cbreak_keyboard_loop,
            args=(stop_event, {" ": _start_new_episode}),
            daemon=True,
        ).start()

    def _record_frames(frames: List[Optional[np.ndarray]]) -> None:
        if not video_writers:
            return
        for cam, frame in zip(cameras, frames):
            if frame is None:
                continue
            writer = video_writers.get(cam.name)
            if writer is None:
                continue
            try:
                writer.write(frame)
                frame_counts[cam.name] = frame_counts.get(cam.name, 0) + 1
            except Exception as exc:
                logging.warning("Failed to write frame for %s: %s", cam.name, exc)

    def _ready_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
        status = "READY — press SPACE to replay"
        cv2.putText(canvas, status, (20, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    def _running_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
        status = f"RUNNING  step {step_count}"
        cv2.putText(canvas, status, (disp_w - 350, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

    def _handle_preview_key(key: int) -> bool:
        if key in (ord('q'), ord('Q'), 27):
            stop_event.set()
            return True
        if key == ord(' '):
            _start_new_episode()
        return False

    try:
        with contextlib.ExitStack() as stack:
            if cameras and args.visualize:
                preview = stack.enter_context(
                    LivePreview(
                        window_name="deploy.py — SPACE=replay, Q=quit",
                        cam_labels=cam_labels,
                        fullscreen=True,
                        wait_key_ms=1,
                    )
                )

            while not stop_event.is_set():
                t_start = time.monotonic()

                # Grab one frame per camera per iteration and record it; reused for
                # observation/visualisation below so everything stays in sync.
                # CameraStabilizer flips any cameras whose feed suddenly matches the
                # previous stabilized frame better when rotated 180° (firmware hiccup).
                frames = stabilizer.process([cam.get_frame() for cam in cameras])
                _record_frames(frames)

                if not running_event.is_set():
                    if preview is not None:
                        key = preview.render(frames, _ready_overlay)
                        if _handle_preview_key(key):
                            break
                    else:
                        time.sleep(0.05)
                    continue

                # 1. Read follower state (observation)
                state_parts_raw: List[np.ndarray] = []
                for key in arm_keys:
                    qpos = state_readers[key].get_joint_pos()
                    if qpos is not None:
                        state_parts_raw.append(qpos)
                if len(state_parts_raw) != len(arm_keys):
                    time.sleep(0.01)
                    continue

                # 1a. Subtract per-episode revolution offset from wrap joints so the
                # policy observes the same wrapped range it was trained on.
                # Offsets are latched on the first state read of each episode and
                # reused for all subsequent steps until SPACE restarts the episode.
                state_parts = [
                    wrap_manager.subtract_state(key, qpos_raw, episode_count)
                    for key, qpos_raw in zip(arm_keys, state_parts_raw)
                ]
                state = np.concatenate(state_parts)
                if policy_type == "diffusion" and expected_state_dim is not None and state.shape[0] != expected_state_dim:
                    logging.error(
                        "Expected a %s-D state vector for DP, but received %s-D. Check the active arm config.",
                        expected_state_dim,
                        state.shape[0],
                    )
                    time.sleep(0.05)
                    continue

                # 2. Read camera images (i-th camera -> policy's i-th VISUAL input key)
                observation: dict = {
                    "observation.state": torch.from_numpy(state),
                }
                for i, cam in enumerate(cameras):
                    bgr = frames[i]
                    feature_key = image_feature_keys[i]
                    if bgr is not None:
                        if resize_transform is None and policy_visual_size is not None:
                            bgr = cv2.resize(bgr, (policy_visual_size[1], policy_visual_size[0]), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                        if resize_transform is not None:
                            img_tensor = resize_transform(img_tensor)
                        observation[feature_key] = img_tensor

                missing_features = [key for key in image_feature_keys if key not in observation]
                if missing_features:
                    logging.error(
                        "Missing observation image(s): %s. Cameras may still be warming up or failing.",
                        missing_features,
                    )
                    time.sleep(0.05)
                    continue

                if policy_type == "diffusion":
                    observation = {key: observation[key] for key in (image_feature_keys + ["observation.state"])}

                # 3. Run policy
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
                        "Policy action dim %s is smaller than expected %s",
                        action_np.size,
                        expected_action_dim,
                    )
                    time.sleep(0.05)
                    continue

                # 4. Send actions to UR arms. Add the latched wrap-joint offset
                # back onto the action so the robot receives absolute joint
                # angles matching its current revolution.
                if running_event.is_set():
                    slice_start = 0
                    for key in arm_keys:
                        joint_count = len(arm_joint_names[key])
                        arm_action = action_np[slice_start:slice_start + joint_count].copy()
                        arm_action = wrap_manager.add_action(key, arm_action)
                        cmd_publishers[key].send(arm_action)
                        slice_start += joint_count

                step_count += 1
                if step_count % 100 == 0:
                    logging.info(f"Step {step_count}")

                # 5. Visualisation
                if preview is not None:
                    key = preview.render(frames, _running_overlay)
                    if _handle_preview_key(key):
                        break

                elapsed = time.monotonic() - t_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except Exception as e:
        logging.error(f"Error in control loop: {e}", exc_info=True)
    finally:
        logging.info(f"Stopping after {step_count} steps")
        for cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass
        for cam_name, writer in video_writers.items():
            try:
                writer.release()
            except Exception:
                pass
            logging.info(
                "Recorded %d frames for %s -> %s",
                frame_counts.get(cam_name, 0),
                cam_name,
                video_paths.get(cam_name, "<unknown>"),
            )
        node.destroy_node()
        rclpy.try_shutdown()
        logging.info("Deploy stopped.")


if __name__ == "__main__":
    main()
