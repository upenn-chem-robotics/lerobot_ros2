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
    docker exec -it ur_robotiq bash -c 'source /ws/install/setup.bash; colcon build; ros2 launch ur_robotiq control_bimanual_ur3_robotiq.launch.py left_robot_ip:=192.168.1.4 right_robot_ip:=192.168.1.5 mode:=none use_gello:=true'


    # Terminal 2: deploy policy
    lerobot-ros-deploy --policy outputs/act_pick_place/checkpoints/last/pretrained_model
    lerobot-ros-deploy --policy outputs/dp_pick_place/checkpoints/last/pretrained_model --visualize
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
from typing import Callable, Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from torchvision.transforms import v2 as transforms_v2

# Reuse the robust homing/reset helpers from record.py so deploy homes the
# arms the same safe way the recorder does: open grippers, skip the home
# publish for arms already at home (republishing a zero-delta trajectory can
# fault the custom UR bridge, e.g. "too high voltage"), then verify.
from lerobot_ros2.cli.record import (
    _HOME_SKIP_TOL_RAD,
    _HOME_VERIFY_TOL_RAD,
    _arm_non_gripper_max_delta,
    _open_gripper_before_home,
)
from lerobot_ros2.config_paths import resolve_config_path
from lerobot_ros2.helper import (
    CameraStabilizer,
    ManualFlip180,
    RobotHomeSender,
    ROSJointReader,
    WrapJointManager,
    _slugify_camera_name,
    extract_home_joint_names,
    extract_home_positions,
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
from lerobot_ros2.visualizer import LivePreview, tile_frames_grid

# A joint state older than this means the UR driver stopped publishing, so the
# cached reading says nothing about where the arm actually is.
_STATE_MAX_AGE_S = 1.0
# Extra window, past send_home's own settle sleeps, for a slow trajectory to
# finish before we call the homing attempt failed.
_HOME_SETTLE_TIMEOUT_S = 15.0
_HOME_MAX_ATTEMPTS = 2

try:
    import rclpy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from lerobot.policies.factory import make_pre_post_processors

from lerobot_ros2.policy_runtime import (
    ROSActionPublisher,
    build_strided_runner,
    detect_fullres_crops,
    detect_per_camera_crops,
    detect_training_resize,
    find_experiment_config_candidates,
    get_image_feature_keys,
    get_policy_visual_size,
    load_policy,
    parse_fullres_crop_args,
    resolve_arm_keys,
    resolve_device,
    resolve_experiment_config_path,
    resolve_recording_config_path,
)
from lerobot_ros2.preprocessing import apply_fullres_crop, apply_per_camera_crop


def _diff_against_training_config(
    policy_path: str, deploy_settings: Dict[str, object]
) -> Dict[str, object]:
    """Return a side-by-side comparison of every relevant policy field.

    Compares each entry in ``deploy_settings`` against the matching field
    inside ``train_config.json`` (which lerobot writes once at the start of
    training and never modifies). NOTE: we deliberately do *not* read
    ``config.json`` here — that file is what gets re-loaded into the policy at
    deploy time, so editing it (e.g. to bump ``n_action_steps`` for a sweep)
    would make ``config.json`` and ``deploy_settings`` identical, hiding the
    very override the user is trying to record.

    Result shape::

        {
          "_summary": "1 of 11 fields differ (n_action_steps)",
          "fields": {
            "n_action_steps":      {"train": 8,  "deploy": 16, "match": false},
            "num_inference_steps": {"train": 10, "deploy": 10, "match": true},
            ...
          }
        }

    If ``train_config.json`` cannot be located, a single ``_warning`` entry is
    returned so the missing reference is visible in the manifest.

    ``hz`` is excluded because it's a deploy-only knob with no training-time
    counterpart. ``policy_type`` is mapped to the saved ``policy.type`` field.
    """
    base = Path(policy_path)
    if base.is_file():
        base = base.parent

    # ``train_config.json`` is normally saved alongside ``config.json`` in the
    # pretrained_model dir, but some older checkpoints stash it one level up
    # (next to the ``pretrained_model`` and ``training_state`` folders).
    candidates = [
        base / "train_config.json",
        base.parent / "train_config.json",
    ]
    train_cfg_path = next((p for p in candidates if p.is_file()), None)
    if train_cfg_path is None:
        logging.warning(
            "No train_config.json found near %s; cannot compute deploy-vs-train diff. "
            "Looked in: %s",
            policy_path,
            ", ".join(str(p) for p in candidates),
        )
        return {
            "_warning": (
                "train_config.json not found next to checkpoint; deploy-vs-train "
                "comparison skipped."
            )
        }
    try:
        train_cfg_full = json.loads(train_cfg_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Could not parse %s: %s", train_cfg_path, exc)
        return {"_warning": f"Failed to parse {train_cfg_path}: {exc}"}

    # The training-time policy hyperparameters live under the ``policy`` block
    # of train_config.json (see scripts/lerobot_train.py).
    train_policy = train_cfg_full.get("policy") or {}

    # Map manifest field name -> key as it appears under train_config.policy.
    field_to_cfg_key = {
        "policy_type": "type",
        "n_obs_steps": "n_obs_steps",
        "n_action_steps": "n_action_steps",
        "horizon": "horizon",
        "num_inference_steps": "num_inference_steps",
        "num_train_timesteps": "num_train_timesteps",
        "noise_scheduler_type": "noise_scheduler_type",
        "prediction_type": "prediction_type",
        "use_amp": "use_amp",
        "vision_backbone": "vision_backbone",
        "device": "device",
    }

    fields: Dict[str, object] = {}
    differing: List[str] = []
    for manifest_key, cfg_key in field_to_cfg_key.items():
        if manifest_key not in deploy_settings:
            continue
        if cfg_key not in train_policy:
            continue
        deploy_val = deploy_settings[manifest_key]
        train_val = train_policy[cfg_key]
        match = deploy_val == train_val
        fields[manifest_key] = {
            "train": train_val,
            "deploy": deploy_val,
            "match": match,
        }
        if not match:
            differing.append(manifest_key)

    if not fields:
        return {"_warning": "No comparable policy fields found in train_config.json."}

    if differing:
        summary = (
            f"{len(differing)} of {len(fields)} fields differ "
            f"({', '.join(differing)})"
        )
    else:
        summary = f"all {len(fields)} fields match training"

    return {"_summary": summary, "fields": fields}


def _summarize_deploy_settings(policy, hz: float) -> Dict[str, object]:
    """Pull the deploy-relevant policy fields out for top-level visibility.

    Anything accessed here lives inside ``policy.config`` (the *currently
    loaded* policy at deploy time), so it reflects exactly what's running —
    even if the user overrode something via CLI before calling
    ``predict_action_chunk``.
    """
    cfg = policy.config

    def _get(name, default=None):
        return getattr(cfg, name, default)

    return {
        "hz": float(hz),
        "policy_type": getattr(cfg, "type", None),
        "n_obs_steps": _get("n_obs_steps"),
        "n_action_steps": _get("n_action_steps"),
        "horizon": _get("horizon"),
        "num_inference_steps": _get("num_inference_steps"),
        "num_train_timesteps": _get("num_train_timesteps"),
        "noise_scheduler_type": _get("noise_scheduler_type"),
        "prediction_type": _get("prediction_type"),
        "use_amp": _get("use_amp"),
        "vision_backbone": _get("vision_backbone"),
        "device": _get("device"),
    }


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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Where to write the deployments/<timestamp>/ rollout videos and "
            "deployment_info.yaml. Defaults to the policy directory, which "
            "mixes runs from a shared checkpoint together; point this at a "
            "per-experiment directory to keep them separate."
        ),
    )
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
    parser.add_argument(
        "--fullres-crop",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "TOP", "LEFT", "HEIGHT", "WIDTH"),
        help=(
            "Override the per-camera full-res crop ROI applied to the raw "
            "camera frame before resizing, e.g. "
            "--fullres-crop right_wrist_top 3 436 704 505. Repeatable. When "
            "omitted, deploy auto-detects the ROIs recorded in the training "
            "dataset's info.json (source_crop)."
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

    device = resolve_device(args.device)

    # ── Load policy ──────────────────────────────────────────────────────
    logging.info(f"Loading policy from {args.policy}...")
    policy, policy_type = load_policy(args.policy, args.deploy_mode, device)
    _, strided_runner = build_strided_runner(policy, policy_type, args.policy, args.hz)
    logging.info(f"Policy loaded: {policy.config.type} on {device}")
    logging.info(f"  Input:  {list(policy.config.input_features.keys())}")
    logging.info(f"  Output: {list(policy.config.output_features.keys())}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
    )

    image_feature_keys = get_image_feature_keys(policy)
    expected_state_dim = policy.config.robot_state_feature.shape[0] if policy.config.robot_state_feature is not None else None
    policy_visual_size = get_policy_visual_size(policy)
    if policy_type in ("diffusion", "action_history_diffusion"):
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
    camera_configs = load_camera_configs(cfg)
    logging.info("Opening configured cameras: %s", [f"{camera.index}:{camera.name}" for camera in camera_configs])
    cameras = open_configured_cameras(camera_configs)
    time.sleep(0.3)
    logging.info(f"{len(cameras)} camera(s) active")

    readers_by_name = {reader.name: reader for reader in cameras}

    # ── Map cameras to policy image keys BY NAME (not by position) ────────
    # The policy's VISUAL input keys are ``observation.images.<slug>`` where
    # ``<slug>`` is the slugified camera name (the ``camera_NN_`` record-time
    # prefix is stripped by downsample). ``cameras`` is sorted alphabetically by
    # name, which need not match the order of ``input_features`` in the
    # checkpoint. Matching by name keeps each physical view on the encoder slot
    # it was trained with regardless of ordering, fixing existing checkpoints.
    camera_index_by_feature_key: Dict[str, int] = {}
    for cam_idx, cam in enumerate(cameras):
        camera_index_by_feature_key[f"observation.images.{_slugify_camera_name(cam.name)}"] = cam_idx
    unmatched_policy_keys = [k for k in image_feature_keys if k not in camera_index_by_feature_key]
    if unmatched_policy_keys:
        logging.error(
            "No camera matches policy image key(s) %s by name. "
            "Available cameras: %s. Policy expects keys: %s. "
            "Rename cameras in %s so each '%s<name>' has a camera named '<name>'.",
            unmatched_policy_keys,
            [cam.name for cam in cameras],
            image_feature_keys,
            config_path,
            "observation.images.",
        )
        sys.exit(1)
    logging.info(
        "Mapping cameras to policy image keys by name: %s",
        {k: cameras[camera_index_by_feature_key[k]].name for k in image_feature_keys},
    )

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
    # The wrist cameras auto-rotate their own image when their gravity sensor
    # decides they are upside down, and stay that way for the rest of the
    # session. Number keys toggle a manual 180° correction so the operator can
    # match the orientation the policy was trained on without replugging.
    manual_flip = ManualFlip180(len(cameras))
    stabilizer = CameraStabilizer(
        stabilize_flags, orientation_mae_thresholds, manual_flip=manual_flip
    )
    for cam, flag, thr in zip(cameras, stabilize_flags, orientation_mae_thresholds):
        if flag:
            logging.info(
                "[%s] orientation-180 stabilization active (mae_delta_threshold=%.1f)",
                cam.name or f"cam{cam.index}",
                thr,
            )

    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        snapshot_dir = Path(args.output_dir)
    else:
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

        # Snapshot of the *currently loaded* policy's deploy-time knobs
        # (n_action_steps, num_inference_steps, etc.) and a compact list of any
        # fields that diverge from the training-time config.json sitting next
        # to the checkpoint. The full config.json/train_config.json live
        # alongside the checkpoint, no need to duplicate them here.
        deploy_settings_summary = _summarize_deploy_settings(policy, args.hz)
        train_diffs = _diff_against_training_config(args.policy, deploy_settings_summary)

        # Manifest for this run, co-located with the videos. This is the sole
        # record of what v4l2 settings actually stuck at deploy time (the
        # record-side equivalent lives in recording_config.yaml).
        manifest = {
            "run_stamp": run_stamp,
            "deployed_at": datetime.now().isoformat(timespec="seconds"),
            "policy_path": str(args.policy),
            "hz": float(args.hz),
            "deploy_settings": deploy_settings_summary,
            "differs_from_training": train_diffs,
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
            "Policy expects these keys (matched by name): %s. Adjust %s or the checkpoint.",
            len(cameras),
            len(image_feature_keys),
            image_feature_keys,
            config_path,
        )
        sys.exit(1)

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
    arm_keys = resolve_arm_keys(arm_configs, arm_joint_names_from_cfg, policy)
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

    # resolve_home() falls back to the live joint state for any arm the config
    # doesn't cover. That fallback is fine for record.py (which is *defining*
    # home) but never for deploy: it makes "home" mean "the pose the arms were
    # in when this process launched", so the first rollout skips homing
    # entirely (zero delta) and every later reset parks at an arbitrary pose.
    config_home_positions = extract_home_positions(experiment_config)
    arms_without_config_home = [k for k in arm_keys if not config_home_positions.get(k)]
    if arms_without_config_home:
        candidates = find_experiment_config_candidates(args.experiment_config, args.policy)
        hint = (
            "Candidate configs that do define a home pose:\n  "
            + "\n  ".join(str(p) for p in candidates)
            if candidates
            else "No config with 'home_positions' was found near the checkpoint. "
            "Re-record or copy the experiment_config.yaml saved with the dataset."
        )
        logging.error(
            "No home pose configured for arm(s) %s (experiment config: %s). Deploy "
            "would otherwise use the arms' current pose as home, which means the "
            "robot never actually homes before a rollout. Pass --experiment-config "
            "pointing at a file with 'home_positions'.\n%s",
            arms_without_config_home,
            experiment_config_path or "<none>",
            hint,
        )
        sys.exit(1)

    try:
        wrap_joint_suffixes = resolve_wrap_joints(cfg)
    except ValueError as exc:
        sys.exit(f"Invalid wrap_joints in recording config: {exc}")
    unwrap_opts = resolve_unwrap_config(cfg)

    def _home_state_source(arm_name: str) -> Optional[np.ndarray]:
        reader = state_readers.get(arm_name)
        if reader is None:
            return None
        # Reject stale samples: a protective stop or a dropped hardware
        # interface freezes the topic, and the last cached reading would then
        # make a motionless arm look like it reached (or is already at) home.
        return reader.get_joint_pos(max_age_s=_STATE_MAX_AGE_S)

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
    if policy_type in ("diffusion", "action_history_diffusion") and expected_state_dim is not None and expected_state_dim != expected_action_dim:
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
    has_run_once = False
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
        f"  Press SPACE -> start rollout (when idle) / stop rollout (when running)\n"
        f"  Press Q     -> quit\n"
        f"  Waiting for SPACE to start...\n"
    )

    preview: Optional[LivePreview] = None
    cam_labels: List[str] = []
    if cameras:
        cam_labels = [c.display_label if getattr(c, "display_label", "") else f"cam {c.index}" for c in cameras]

    terminal_keyboard_enabled = not args.visualize

    # ── Per-episode grid recording ───────────────────────────────────────
    # Each SPACE press opens a fresh grid_episode_<NNN>.mp4 alongside the
    # per-camera MP4s. The grid stitches all stabilized cameras into one
    # frame at args.hz so the full multi-view timeline of a single episode
    # lives in a single file (handy for slides / side-by-side review).
    grid_columns = max(1, int((cfg.get("visualization") or {}).get("camera_grid_columns") or 2))
    grid_rows = int(np.ceil(len(cameras) / grid_columns)) if cameras else 0
    grid_tile_w = max((int(c.width) for c in cameras), default=0)
    grid_tile_h = max((int(c.height) for c in cameras), default=0)
    grid_target_w = grid_columns * grid_tile_w
    grid_target_h = grid_rows * grid_tile_h
    grid_writer: Optional[cv2.VideoWriter] = None
    grid_video_path: Optional[Path] = None
    grid_frame_count: int = 0
    grid_labels = [c.name or f"cam{c.index}" for c in cameras]

    def _close_episode_grid_writer() -> None:
        nonlocal grid_writer, grid_video_path, grid_frame_count
        if grid_writer is None:
            return
        try:
            grid_writer.release()
        except Exception:
            pass
        logging.info(
            "Recorded %d grid frames -> %s",
            grid_frame_count,
            grid_video_path,
        )
        grid_writer = None
        grid_video_path = None
        grid_frame_count = 0

    def _open_episode_grid_writer(episode_idx: int) -> None:
        nonlocal grid_writer, grid_video_path, grid_frame_count
        if (
            deploy_run_dir is None
            or not cameras
            or grid_target_w <= 0
            or grid_target_h <= 0
        ):
            return
        path = deploy_run_dir / f"grid_episode_{episode_idx:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            float(args.hz),
            (grid_target_w, grid_target_h),
        )
        if not writer.isOpened():
            logging.warning(
                "Could not open grid VideoWriter at %s (size=%sx%s, fps=%s); "
                "skipping grid for this episode.",
                path,
                grid_target_w,
                grid_target_h,
                args.hz,
            )
            try:
                writer.release()
            except Exception:
                pass
            return
        grid_writer = writer
        grid_video_path = path
        grid_frame_count = 0
        logging.info(
            "Recording grid -> %s (%sx%s @ %s Hz)",
            path,
            grid_target_w,
            grid_target_h,
            args.hz,
        )

    def _wait_for_arms_at_home(arms: List[str], timeout_s: float) -> Dict[str, Optional[float]]:
        """Poll until every arm in ``arms`` is within the verify tolerance.

        Returns the last measured per-arm delta (``None`` when the state feed
        went stale). Exits as soon as all arms are home, so this costs nothing
        in the normal case and only burns wall-clock on a slow trajectory.
        """
        deadline = time.monotonic() + timeout_s
        deltas: Dict[str, Optional[float]] = {}
        while True:
            deltas = {
                arm: _arm_non_gripper_max_delta(
                    arm_configs, home_positions, arm_joint_names, _home_state_source, arm
                )
                for arm in arms
            }
            if all(d is not None and d <= _HOME_VERIFY_TOL_RAD for d in deltas.values()):
                return deltas
            if time.monotonic() >= deadline:
                return deltas
            time.sleep(0.1)

    def _robust_home() -> bool:
        """Home the arms safely (mirrors record.py's reset sequence).

        1. open right gripper, then left gripper
        2. home only arms that aren't already essentially at home — republishing
           a zero-delta home trajectory can fault the custom UR bridge (e.g.
           "too high voltage") and require a controller restart
        3. verify each arm ended up at home, retrying once before giving up

        Returns True only when every arm is confirmed at home; the caller must
        not start the policy otherwise.
        """
        home_order = [a for a in ("right", "left") if a in home_positions]

        for attempt in range(1, _HOME_MAX_ATTEMPTS + 1):
            for arm in home_order:
                _open_gripper_before_home(
                    arm_configs,
                    home_positions,
                    arm_joint_names,
                    home_sender,
                    _home_state_source,
                    arm,
                    label="deploy",
                )

            homing_arms: List[str] = []
            for arm in home_order:
                delta = _arm_non_gripper_max_delta(
                    arm_configs, home_positions, arm_joint_names, _home_state_source, arm
                )
                if delta is None:
                    # State unavailable — be safe and publish.
                    logging.warning(
                        "[deploy] [%s] no fresh joint state (topic stale for >%.1fs); "
                        "publishing home anyway",
                        arm, _STATE_MAX_AGE_S,
                    )
                    homing_arms.append(arm)
                    continue
                if delta < _HOME_SKIP_TOL_RAD:
                    logging.info(
                        "[deploy] [%s] already at home (max_delta=%.4f rad); "
                        "skipping home publish",
                        arm, delta,
                    )
                    continue
                homing_arms.append(arm)

            if homing_arms:
                logging.info(
                    "[deploy] homing %s (attempt %d/%d)",
                    homing_arms, attempt, _HOME_MAX_ATTEMPTS,
                )
                ordered_home_positions = {a: home_positions[a] for a in homing_arms}
                ordered_home_joint_names = {a: arm_joint_names[a] for a in homing_arms}
                home_sender.send_home(ordered_home_positions, ordered_home_joint_names)

            # send_home's fixed settle sleeps assume the trajectory finishes in
            # time; give a slow arm a bounded extra window rather than
            # declaring failure the instant the sleep expires.
            deltas = _wait_for_arms_at_home(home_order, _HOME_SETTLE_TIMEOUT_S)

            stuck: List[str] = []
            for arm in home_order:
                delta = deltas.get(arm)
                if delta is None:
                    stuck.append(arm)
                    logging.error(
                        "[deploy] [%s] cannot verify home — no joint state for "
                        "more than %.1fs. The UR driver has likely stopped "
                        "publishing (protective stop or dead hardware interface).",
                        arm, _STATE_MAX_AGE_S,
                    )
                elif delta > _HOME_VERIFY_TOL_RAD:
                    stuck.append(arm)
                    logging.error(
                        "[deploy] [%s] DID NOT REACH HOME after homing "
                        "(max_delta=%.3f rad, tol=%.3f). The arm controller likely "
                        "did not execute the trajectory — possible protective stop, "
                        "fault (e.g. too high voltage), inactive controller, or dead "
                        "hardware interface. Check the ur_robotiq container logs and "
                        "`ros2 control list_hardware_components`. A driver restart is "
                        "likely required.",
                        arm, delta, _HOME_VERIFY_TOL_RAD,
                    )
                else:
                    logging.info("[deploy] [%s] at home (max_delta=%.4f rad)", arm, delta)

            if not stuck:
                return True
            if attempt < _HOME_MAX_ATTEMPTS:
                logging.warning(
                    "[deploy] retrying home for %s (attempt %d/%d)",
                    stuck, attempt + 1, _HOME_MAX_ATTEMPTS,
                )

        logging.error(
            "[deploy] Homing FAILED after %d attempt(s); the policy will NOT be "
            "started. Recover the arm controller, then press SPACE again.",
            _HOME_MAX_ATTEMPTS,
        )
        return False

    def _start_new_episode() -> None:
        nonlocal episode_count, has_run_once
        running_event.clear()
        # Close the previous episode's grid writer BEFORE homing so the
        # homing motion (which can take >10s while send_home blocks) does
        # not get appended to the prior episode's video. The main control
        # loop keeps calling _record_frames during the blocking home_sender
        # call, so leaving the writer open here would include homing.
        _close_episode_grid_writer()
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
        if not _robust_home():
            # Starting a rollout from a pose that isn't home means the policy
            # runs from an initial state it was never trained on, so leave the
            # loop idle and let the operator recover the controller first.
            logging.error("REPLAY — NOT starting policy: arms are not at home")
            return
        episode_count += 1
        _open_episode_grid_writer(episode_count)
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
        has_run_once = True
        logging.info(f"REPLAY — policy running (episode {episode_count})")

    def _toggle_policy() -> None:
        # SPACE acts as a two-stage toggle: stop a running rollout first so
        # the operator can prepare, then start a fresh rollout on the next
        # press. Resets happen inside _start_new_episode, so simply clearing
        # running_event here is enough to halt the loop (the control loop's
        # `if running_event.is_set(): cmd_publishers[...].send(...)` guard
        # leaves the robot holding its current pose).
        if running_event.is_set():
            running_event.clear()
            # Finalize the current grid mp4 immediately so the file on disk
            # ends at the stop press instead of trailing on with idle frames
            # until the next rollout starts. _record_frames is also gated on
            # running_event below, so per-camera writers stop receiving
            # frames at the same instant.
            _close_episode_grid_writer()
            logging.info("STOPPED — press SPACE to start next rollout")
        else:
            _start_new_episode()

    def _record_frames(frames: List[Optional[np.ndarray]]) -> None:
        nonlocal grid_frame_count
        # Only record while a rollout is actively running. Previously the
        # per-camera mp4s and the grid mp4 kept accepting frames in the
        # interval between SPACE-stop and SPACE-start (and during the
        # homing transition), which padded each deployment with idle
        # footage. Gating here makes recording match policy execution.
        if not running_event.is_set():
            return
        if not video_writers and grid_writer is None:
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
        if grid_writer is not None and cameras:
            try:
                tile = tile_frames_grid(
                    frames,
                    grid_labels,
                    grid_target_w,
                    grid_target_h,
                    grid_columns,
                )
                grid_writer.write(tile)
                grid_frame_count += 1
            except Exception as exc:
                logging.warning("Failed to write grid frame: %s", exc)

    def _ready_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
        status = "READY — press SPACE to start"
        cv2.putText(canvas, status, (20, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    def _stopped_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
        status = "STOPPED — press SPACE to start next rollout"
        cv2.putText(canvas, status, (20, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    def _running_overlay(canvas: np.ndarray, disp_w: int, disp_h: int) -> None:
        status = f"RUNNING  step {step_count}"
        cv2.putText(canvas, status, (disp_w - 350, disp_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

    def _flip_annotated_labels() -> List[str]:
        return [
            f"{label}  [FLIP180]" if manual_flip.is_flipped(i) else label
            for i, label in enumerate(cam_labels)
        ]

    def _toggle_camera_flip(slot: int) -> None:
        new_state = manual_flip.toggle(slot)
        if new_state is None:
            return
        cam = cameras[slot]
        logging.warning(
            "[%s] manual 180° flip %s — frames fed to the policy are now %s",
            cam.name or f"cam{cam.index}",
            "ON" if new_state else "OFF",
            "rotated" if new_state else "as-captured",
        )
        if preview is not None:
            preview.set_labels(_flip_annotated_labels())

    def _handle_preview_key(key: int) -> bool:
        if key in (ord('q'), ord('Q'), 27):
            stop_event.set()
            return True
        if key == ord(' '):
            _toggle_policy()
        elif ord('1') <= key <= ord('9'):
            _toggle_camera_flip(key - ord('1'))
        return False

    # Started here rather than next to `terminal_keyboard_enabled` so the
    # handlers it dispatches to are already defined when the thread starts.
    if terminal_keyboard_enabled:
        terminal_handlers: Dict[str, Callable[[], None]] = {" ": _toggle_policy}
        for _slot in range(min(len(cameras), 9)):
            terminal_handlers[str(_slot + 1)] = (
                lambda slot=_slot: _toggle_camera_flip(slot)
            )
        threading.Thread(
            target=run_cbreak_keyboard_loop,
            args=(stop_event, terminal_handlers),
            daemon=True,
        ).start()

    try:
        with contextlib.ExitStack() as stack:
            if cameras and args.visualize:
                preview = stack.enter_context(
                    LivePreview(
                        window_name="deploy.py — SPACE=start/stop, 1-9=flip cam 180°, Q=quit",
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
                        overlay = _stopped_overlay if has_run_once else _ready_overlay
                        key = preview.render(frames, overlay)
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
                if policy_type in ("diffusion", "action_history_diffusion") and expected_state_dim is not None and state.shape[0] != expected_state_dim:
                    logging.error(
                        "Expected a %s-D state vector for DP, but received %s-D. Check the active arm config.",
                        expected_state_dim,
                        state.shape[0],
                    )
                    time.sleep(0.05)
                    continue

                # 2. Read camera images (matched to policy VISUAL keys BY NAME)
                observation: dict = {
                    "observation.state": torch.from_numpy(state),
                }
                for feature_key in image_feature_keys:
                    bgr = frames[camera_index_by_feature_key[feature_key]]
                    if bgr is not None:
                        if fullres_crops:
                            bgr = apply_fullres_crop(bgr, feature_key, fullres_crops)
                        if resize_transform is None and policy_visual_size is not None:
                            bgr = cv2.resize(bgr, (policy_visual_size[1], policy_visual_size[0]), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                        if resize_transform is not None:
                            img_tensor = resize_transform(img_tensor)
                        if per_camera_crops:
                            img_tensor = apply_per_camera_crop(
                                img_tensor, feature_key, per_camera_crops
                            )
                        observation[feature_key] = img_tensor

                missing_features = [key for key in image_feature_keys if key not in observation]
                if missing_features:
                    logging.error(
                        "Missing observation image(s): %s. Cameras may still be warming up or failing.",
                        missing_features,
                    )
                    time.sleep(0.05)
                    continue

                if policy_type in ("diffusion", "action_history_diffusion"):
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
        _close_episode_grid_writer()
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
