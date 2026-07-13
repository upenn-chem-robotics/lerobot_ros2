#!/usr/bin/env python3
"""Render an MP4 that shows plateau detection on top of episode video.

For each requested episode this CLI builds a side-by-side composite:

* Top: every camera tile from the dataset's mp4(s), tiled horizontally.
* Bottom: a thin timeline strip with one column per frame, colored
  green = anchor-kept, red = plateau-flagged (would be dropped),
  blue = pre-existing ``action_source == 0`` (e.g. DAgger prefix).
* A small status line in the top-left shows the current frame's
  classification, the normalized action speed ``s_t``, and the
  hyperparameters used for detection.

The MP4 is written at the dataset's native FPS so playback shows the
real cadence of pauses. **Nothing in the source dataset is modified.**

Pair this with ``lerobot-ros-plateau-stats`` -- use the stats CLI to
explore thresholds globally, then this CLI to verify on a handful of
episodes that the red bands really do match the operator's hesitations
(and not, e.g., the slow pour itself).

Examples::

    # One episode
    lerobot-ros-plateau-visualize \
        --src     /lerobot-ros/data/pour_20260514_merged_ds5 \
        --episode 0 \
        --output  /tmp/ep000_plateau.mp4

    # A handful of episodes to ``out_dir/episode_XXXX_plateau.mp4``
    lerobot-ros-plateau-visualize \
        --src     /lerobot-ros/data/pour_20260514_merged_ds5 \
        --episodes 0,7,42,97 \
        --output-dir /tmp/plateau_check

    # Visualize tighter thresholds without touching the dataset
    lerobot-ros-plateau-visualize \
        --src /lerobot-ros/data/pour_20260514_merged_ds5 \
        --episode 0 --output /tmp/ep000.mp4 \
        --tau 0.003 --min-run 8 --margin 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import av  # type: ignore[import-not-found]
import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot_ros2.plateau import (
    PlateauParams,
    PlateauResult,
    detect_plateaus,
    load_action_min_max_from_stats,
)
from lerobot_ros2.video_exporter import _open_video_writer

_GREEN = (60, 220, 60)   # BGR -- anchor kept
_RED = (60, 60, 220)     # BGR -- plateau flagged
_BLUE = (220, 140, 60)   # BGR -- pre-existing action_source == 0
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _annotate(frame: np.ndarray, lines: list[str], color: tuple[int, int, int] = _WHITE) -> np.ndarray:
    """Draw labeled lines in the top-left corner of a BGR frame."""
    out = frame.copy()
    y = 14
    for line in lines:
        cv2.putText(out, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _BLACK, 2, cv2.LINE_AA)
        cv2.putText(out, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 16
    return out


def _load_action_episode_existing_as(src: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read ``action``, ``episode_index``, and optional ``action_source`` from parquets."""
    data_dir = src / "data"
    parts = sorted(data_dir.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquets under {data_dir}")
    actions: list[np.ndarray] = []
    eps: list[np.ndarray] = []
    a_sources: list[np.ndarray] | None = [] if "action_source" in pq.read_schema(parts[0]).names else None
    for p in parts:
        cols = ["action", "episode_index"]
        if a_sources is not None:
            cols.append("action_source")
        t = pq.read_table(p, columns=cols)
        actions.append(np.stack(t.column("action").to_pylist()).astype(np.float64))
        eps.append(np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64))
        if a_sources is not None:
            raw = t.column("action_source").to_pylist()
            arr = np.asarray([v[0] if isinstance(v, (list, tuple)) else v for v in raw], dtype=np.int64)
            a_sources.append(arr)
    action = np.concatenate(actions, axis=0)
    ep_idx = np.concatenate(eps, axis=0)
    existing_as = np.concatenate(a_sources, axis=0) if a_sources is not None else None
    return action, ep_idx, existing_as


def _episode_video_paths(src: Path, episode_index: int) -> dict[str, Path]:
    """Return ``{camera_key: mp4_path}`` for one episode.

    LeRobot v3 datasets store one mp4 per camera per chunk, with the
    episode boundaries declared via ``from_timestamp``/``to_timestamp``
    in ``meta/episodes/.../*.parquet``. Most of our datasets fit one
    chunk per camera, so we just look up the right mp4 per camera.
    """
    episodes_meta = src / "meta" / "episodes"
    parts = sorted(episodes_meta.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No episode meta parquets under {episodes_meta}")
    info = json.loads((src / "meta" / "info.json").read_text())
    video_path_tmpl = info["video_path"]
    cameras = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    out: dict[str, Path] = {}
    found = False
    for p in parts:
        t = pq.read_table(p).to_pandas()
        row = t[t["episode_index"] == int(episode_index)]
        if row.empty:
            continue
        found = True
        for cam in cameras:
            ci = int(row[f"videos/{cam}/chunk_index"].iloc[0])
            fi = int(row[f"videos/{cam}/file_index"].iloc[0])
            rel = video_path_tmpl.format(video_key=cam, chunk_index=ci, file_index=fi)
            out[cam] = src / rel
        break
    if not found:
        raise ValueError(f"Episode {episode_index} not found under {episodes_meta}")
    return out


def _read_clip_for_episode(
    mp4_path: Path, from_ts: float, to_ts: float, expected_frames: int, fps: int,
) -> np.ndarray:
    """Decode ``[from_ts, to_ts)`` of an mp4 into ``(N, H, W, 3)`` BGR frames.

    Uses PyAV instead of ``cv2.VideoCapture`` because OpenCV's bundled
    ffmpeg backend in this container is missing an H.264 decoder; PyAV
    ships with one and is already used by :mod:`lerobot_ros2.video_exporter`
    for writing.
    """
    frames: list[np.ndarray] = []
    container = av.open(str(mp4_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base if stream.time_base else None
        try:
            seek_pts = int(from_ts / float(tb)) if tb else int(from_ts * float(fps))
        except (TypeError, ZeroDivisionError):
            seek_pts = 0
        try:
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        except av.AVError:
            container.seek(0)
        for frame in container.decode(stream):
            pts_sec = float(frame.pts * tb) if (tb and frame.pts is not None) else None
            if pts_sec is not None and pts_sec + 1e-6 < from_ts:
                continue
            if pts_sec is not None and pts_sec >= to_ts:
                break
            arr = frame.to_ndarray(format="bgr24")
            frames.append(arr)
            if len(frames) >= expected_frames:
                break
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {mp4_path} starting at {from_ts}s")
    if len(frames) < expected_frames:
        last = frames[-1]
        while len(frames) < expected_frames:
            frames.append(last.copy())
    return np.stack(frames[:expected_frames], axis=0)


def _tile_horizontally(frames: list[np.ndarray]) -> np.ndarray:
    """Tile a list of equal-height BGR frames horizontally."""
    target_h = max(f.shape[0] for f in frames)
    resized: list[np.ndarray] = []
    for f in frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = max(1, int(round(f.shape[1] * scale)))
            resized.append(cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA))
        else:
            resized.append(f)
    return np.concatenate(resized, axis=1)


def _build_timeline_strip(
    width: int,
    classes: np.ndarray,
    current_idx: int,
    strip_height: int = 18,
) -> np.ndarray:
    """Build a per-frame colored strip showing keep/plateau/prefix classification.

    ``classes`` is an int array of length ``N``:
      0 = anchor kept (green)
      1 = plateau flagged (red)
      2 = pre-existing action_source==0 (blue)
    """
    n = classes.shape[0]
    if n == 0 or width <= 0:
        return np.zeros((strip_height, max(width, 1), 3), dtype=np.uint8)
    strip = np.zeros((strip_height, width, 3), dtype=np.uint8)
    palette = {0: _GREEN, 1: _RED, 2: _BLUE}
    for j in range(width):
        f0 = (j * n) // width
        f1 = max(f0 + 1, ((j + 1) * n) // width)
        window = classes[f0:f1]
        if (window == 1).any():
            color = _RED
        elif (window == 2).any():
            color = _BLUE
        else:
            color = palette[int(window[0]) if window.size else 0]
        strip[:, j] = color
    cursor = int(round(current_idx / max(n - 1, 1) * (width - 1)))
    cv2.line(strip, (cursor, 0), (cursor, strip_height - 1), _WHITE, 1)
    return strip


# Distinct per-joint colors for the per-joint action plot. BGR (OpenCV order).
# Up to 8 joints supported; cycles if more. Chosen to be visually distinct on a
# dark gray background and to avoid clashing with the speed-plot greens/reds.
_JOINT_PALETTE = [
    (60, 220, 60),     # bright green
    (60, 140, 255),    # orange
    (255, 130, 60),    # cyan-blue
    (200, 80, 255),    # magenta
    (60, 220, 220),    # yellow
    (255, 200, 80),    # sky blue
    (140, 140, 140),   # gray (gripper-ish)
    (80, 200, 255),    # amber
]


def _build_joints_strip(
    width: int,
    action_norm_ep: np.ndarray,
    current_idx: int,
    strip_height: int,
    joint_names: list[str] | None = None,
) -> np.ndarray:
    """Per-joint normalized-action curves over the episode (y in ``[0, 1]``).

    Each joint is drawn as its own colored polyline at full y-range so the
    visual link between "this joint moved" and the L\u221e speed below is
    obvious: if you see *any* colored line take a vertical jump between
    two adjacent frames, that's exactly what contributes to ``s_t``.

    A small legend at the top maps colors to short joint labels (last
    token of the joint name, e.g. ``shoulder_pan``).
    """
    n, d = action_norm_ep.shape
    strip = np.full((strip_height, width, 3), 25, dtype=np.uint8)
    if n == 0 or width <= 0 or d == 0:
        return strip

    cv2.line(strip, (0, 0), (width - 1, 0), (60, 60, 60), 1)
    cv2.line(strip, (0, strip_height - 1), (width - 1, strip_height - 1), (60, 60, 60), 1)
    mid = strip_height - 1 - int(0.5 * (strip_height - 2))
    for x in range(0, width, 6):
        cv2.line(strip, (x, mid), (x + 2, mid), (50, 50, 50), 1)

    for j in range(d):
        color = _JOINT_PALETTE[j % len(_JOINT_PALETTE)]
        pts = np.empty((width, 1, 2), dtype=np.int32)
        for x in range(width):
            f = min(int(x * n / width), n - 1)
            v = float(action_norm_ep[f, j])
            v = max(0.0, min(1.0, v))
            y = strip_height - 1 - int(v * (strip_height - 2))
            pts[x, 0, 0] = x
            pts[x, 0, 1] = int(np.clip(y, 0, strip_height - 1))
        cv2.polylines(strip, [pts], False, color, 1, cv2.LINE_AA)

    cursor = int(round(current_idx / max(n - 1, 1) * (width - 1)))
    cv2.line(strip, (cursor, 0), (cursor, strip_height - 1), _WHITE, 1)

    cv2.putText(strip, "action_norm [0..1] per joint",
                (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, _WHITE, 1, cv2.LINE_AA)
    if joint_names is not None and len(joint_names) >= d:
        legend_x = 4
        legend_y = 26
        for j in range(d):
            color = _JOINT_PALETTE[j % len(_JOINT_PALETTE)]
            short = joint_names[j].rsplit("_joint", 1)[0]
            label = f"j{j}:{short[:14]}"
            cv2.rectangle(strip, (legend_x, legend_y - 6),
                          (legend_x + 8, legend_y + 2), color, -1)
            cv2.putText(strip, label, (legend_x + 12, legend_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            legend_x += 12 + 8 + 6 * len(label)
            if legend_x > width - 80:
                legend_x = 4
                legend_y += 12
                if legend_y > strip_height - 4:
                    break

    cv2.putText(strip, "1.0", (width - 28, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(strip, "0.0", (width - 28, strip_height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
    return strip


_SPEED_PLOT_Y_MAX = 0.01


def _speed_plot_ticks(y_max: float) -> tuple[float, ...]:
    """Quarter marks for the fixed speed-plot y-axis."""
    return tuple((y_max / 4.0) * i for i in range(1, 4))


def _format_speed_axis(value: float) -> str:
    if value < 0.01:
        return f"{value:.4f}"
    if value < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _speed_to_y(speed_value: float, y_max: float, strip_height: int) -> int:
    """Map a normalized speed value onto strip row coordinates."""
    frac = max(0.0, min(1.0, float(speed_value) / y_max))
    return int(np.clip(strip_height - 1 - int(frac * (strip_height - 2)), 0, strip_height - 1))


def _build_speed_strip(
    width: int,
    speed: np.ndarray,
    threshold: float,
    current_idx: int,
    strip_height: int = 80,
    y_max: float = _SPEED_PLOT_Y_MAX,
) -> np.ndarray:
    """Polyline of L\u221e speed ``s_t`` on a fixed ``[0, y_max]`` axis.

    The red horizontal line is drawn at ``y = tau``. Speed values above
    ``y_max`` clip at the top of the plot.
    """
    n = speed.shape[0]
    strip = np.full((strip_height, width, 3), 30, dtype=np.uint8)
    if n == 0 or width <= 0 or y_max <= 0:
        return strip

    def _draw_hgrid(value: float, color: tuple[int, int, int], label: str) -> None:
        gy = _speed_to_y(value, y_max, strip_height)
        for x in range(0, width, 6):
            cv2.line(strip, (x, gy), (x + 2, gy), color, 1)
        cv2.putText(
            strip, label, (width - 44, max(10, min(gy + 4, strip_height - 4))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
        )

    for tick in _speed_plot_ticks(y_max):
        _draw_hgrid(tick, (55, 55, 55), _format_speed_axis(tick))

    pts = np.empty((width, 1, 2), dtype=np.int32)
    for x in range(width):
        f = min(int(x * n / width), n - 1)
        y = _speed_to_y(float(speed[f]), y_max, strip_height)
        pts[x, 0, 0] = x
        pts[x, 0, 1] = y
    cv2.polylines(strip, [pts], False, _GREEN, 1, cv2.LINE_AA)

    thr_y = _speed_to_y(threshold, y_max, strip_height)
    cv2.line(strip, (0, thr_y), (width - 1, thr_y), _RED, 1)
    cv2.putText(
        strip, f"tau={threshold:.4f}", (4, max(10, thr_y - 2)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.32, _RED, 1, cv2.LINE_AA,
    )

    cursor = int(round(current_idx / max(n - 1, 1) * (width - 1)))
    cv2.line(strip, (cursor, 0), (cursor, strip_height - 1), _WHITE, 1)

    cv2.putText(
        strip, f"s_t  y: 0 .. {_format_speed_axis(y_max)}", (4, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.36, _WHITE, 1, cv2.LINE_AA,
    )
    cv2.putText(
        strip, _format_speed_axis(y_max), (width - 44, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA,
    )
    cv2.putText(
        strip, "0.0000", (width - 44, strip_height - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA,
    )
    return strip


def _classes_for_episode(
    ep_slice: slice, plateau_mask: np.ndarray, existing_as: np.ndarray | None,
) -> np.ndarray:
    """0=keep, 1=plateau, 2=pre-existing action_source==0."""
    seg_plateau = plateau_mask[ep_slice]
    seg_existing = (existing_as[ep_slice] == 0) if existing_as is not None else np.zeros_like(seg_plateau)
    out = np.zeros_like(seg_plateau, dtype=np.int8)
    out[seg_existing] = 2
    out[seg_plateau & ~seg_existing] = 1
    return out


def _render_episode(
    src: Path,
    episode_index: int,
    plateau_result: PlateauResult,
    existing_as: np.ndarray | None,
    ep_idx_all: np.ndarray,
    fps: int,
    out_path: Path,
    upscale: int,
    y_max: float = _SPEED_PLOT_Y_MAX,
    joint_names: list[str] | None = None,
    drop_plateau: bool = False,
) -> int:
    """Render one episode to ``out_path``. Returns number of frames written.

    When ``drop_plateau`` is set, frames classified as plateau (no motion)
    are skipped entirely so the output video shows only the moving frames.
    """
    ep_mask = ep_idx_all == int(episode_index)
    ep_positions = np.flatnonzero(ep_mask)
    if ep_positions.size == 0:
        logging.warning("Episode %d not present in dataset; skipping", episode_index)
        return 0
    start, end = int(ep_positions[0]), int(ep_positions[-1] + 1)
    expected_n = end - start

    video_paths = _episode_video_paths(src, episode_index)
    episodes_meta = src / "meta" / "episodes"
    parts = sorted(episodes_meta.rglob("*.parquet"))
    from_ts_per_cam: dict[str, float] = {}
    to_ts_per_cam: dict[str, float] = {}
    for p in parts:
        df = pq.read_table(p).to_pandas()
        row = df[df["episode_index"] == int(episode_index)]
        if row.empty:
            continue
        for cam in video_paths:
            from_ts_per_cam[cam] = float(row[f"videos/{cam}/from_timestamp"].iloc[0])
            to_ts_per_cam[cam] = float(row[f"videos/{cam}/to_timestamp"].iloc[0])
        break

    decoded: dict[str, np.ndarray] = {}
    for cam, mp4 in video_paths.items():
        decoded[cam] = _read_clip_for_episode(
            mp4_path=mp4,
            from_ts=from_ts_per_cam[cam],
            to_ts=to_ts_per_cam[cam],
            expected_frames=expected_n,
            fps=fps,
        )

    classes = _classes_for_episode(slice(start, end), plateau_result.plateau_mask, existing_as)
    ep_speed = plateau_result.speed[start:end]
    ep_action_norm = plateau_result.action_norm[start:end]
    tau = plateau_result.params.tau
    min_run = plateau_result.params.min_run
    margin = plateau_result.params.margin
    plot_y_max = float(y_max)

    cam_keys = sorted(video_paths.keys())
    sample_tile = decoded[cam_keys[0]][0]
    tile_h, tile_w = sample_tile.shape[:2]
    composite_w = tile_w * len(cam_keys)
    composite_h = tile_h

    strip_h = 18
    joints_h = 110
    speed_h = 90
    total_h = composite_h + strip_h + joints_h + speed_h

    up = max(1, int(upscale))
    out_w = composite_w * up
    out_h = total_h * up
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(out_path, out_w, out_h, fps)

    color_for_class = {0: _GREEN, 1: _RED, 2: _BLUE}
    label_for_class = {0: "KEEP", 1: "PLATEAU", 2: "AS=0 (existing)"}

    n_plateau_frames = int((classes == 1).sum())
    n_existing_drop = int((classes == 2).sum())

    written = 0
    try:
        for i in range(expected_n):
            cls = int(classes[i])
            if drop_plateau and cls == 1:
                continue
            tiles = [decoded[c][i] for c in cam_keys]
            top = _tile_horizontally(tiles)
            speed_i = float(ep_speed[i]) if i < ep_speed.size else 0.0
            status_color = color_for_class[cls]
            lines = [
                f"ep {episode_index}  frame {i+1}/{expected_n}",
                f"{label_for_class[cls]}  s_t={speed_i:.4f}  tau={tau:.4f}",
                f"min_run={min_run}f  margin={margin}f  norm={plateau_result.params.norm}",
                f"this ep: plateau={n_plateau_frames}  existing_drop={n_existing_drop}",
            ]
            top = _annotate(top, lines, color=status_color)
            timeline = _build_timeline_strip(composite_w, classes, i, strip_height=strip_h)
            joints_strip = _build_joints_strip(
                composite_w, ep_action_norm, i,
                strip_height=joints_h, joint_names=joint_names,
            )
            speed_strip = _build_speed_strip(
                composite_w, ep_speed, tau, i,
                strip_height=speed_h, y_max=plot_y_max,
            )
            canvas = np.concatenate([top, timeline, joints_strip, speed_strip], axis=0)
            if up > 1:
                canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            writer.write(np.ascontiguousarray(canvas))
            written += 1
    finally:
        writer.release()
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="LeRobot dataset root.")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--episode", type=int, help="Single episode index to render.")
    target.add_argument("--episodes", type=str, help="Comma-separated episode indices, e.g. '0,7,42'.")
    target.add_argument("--all-episodes", action="store_true",
                        help="Render every episode in the dataset.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output mp4 path (required when --episode is set).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (required when --episodes or --all-episodes is set).")
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--min-run", type=int, default=5)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--norm", choices=("linf", "l2"), default="linf")
    p.add_argument("--joints", type=str, default=None,
                   help="Optional comma-separated joint indices to consider.")
    p.add_argument("--upscale", type=int, default=2,
                   help="Nearest-neighbour upscale of the output video (default: 2).")
    p.add_argument("--drop-plateau", action="store_true",
                   help=(
                       "Skip plateau (zero-motion) frames in the output video so "
                       "only the moving frames remain. The bottom strips/plots still "
                       "show the full episode for reference."
                   ))
    p.add_argument(
        "--y-max",
        type=float,
        default=_SPEED_PLOT_Y_MAX,
        help=(
            "Fixed y-axis upper bound for the s_t speed plot (default: 0.01). "
            "The red tau line is drawn at y=tau on this scale; values above "
            "y-max clip at the top. Must be greater than --tau."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Speed plot y-axis: 0 .. %s (tau=%s)", args.y_max, args.tau)

    if float(args.tau) >= float(args.y_max):
        logging.warning(
            "--tau (%.4f) >= --y-max (%.4f); the red threshold line will clip "
            "at the top of the speed plot.",
            args.tau,
            args.y_max,
        )

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"error: --src does not exist: {src}", file=sys.stderr)
        return 2

    if args.episode is not None and not args.output:
        print("error: --output required when --episode is set", file=sys.stderr)
        return 2
    if (args.episodes or args.all_episodes) and not args.output_dir:
        print("error: --output-dir required when --episodes or --all-episodes is set", file=sys.stderr)
        return 2

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = int(info.get("fps", 10))
    action_feature = (info.get("features") or {}).get("action") or {}
    joint_names = list(action_feature.get("names") or [])

    a_min, a_max = load_action_min_max_from_stats(src / "meta" / "stats.json")
    action, ep_idx, existing_as = _load_action_episode_existing_as(src)

    joints = None
    if args.joints:
        joints = tuple(int(x) for x in args.joints.split(","))

    params = PlateauParams(
        tau=float(args.tau), min_run=int(args.min_run), margin=int(args.margin),
        norm=args.norm, joint_indices=joints,
    )
    logging.info(
        "Running plateau detection: tau=%s min_run=%s margin=%s norm=%s joints=%s",
        params.tau, params.min_run, params.margin, params.norm, params.joint_indices,
    )
    result = detect_plateaus(action, ep_idx, a_min, a_max, params)
    s = result.stats
    logging.info(
        "Detected %d / %d plateau frames (%.2f%%) across %d episodes",
        s["n_dropped_frames"], s["n_total_frames"],
        100 * s["n_dropped_frames"] / max(s["n_total_frames"], 1),
        s["n_episodes"],
    )

    if args.episode is not None:
        episodes = [int(args.episode)]
        out_paths = {episodes[0]: args.output.expanduser().resolve()}
    else:
        if args.all_episodes:
            episodes = sorted(int(e) for e in np.unique(ep_idx))
        else:
            episodes = [int(x) for x in args.episodes.split(",")]
        out_dir = args.output_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = {e: out_dir / f"episode_{e:04d}_plateau.mp4" for e in episodes}

    for ep in episodes:
        out = out_paths[ep]
        logging.info("Rendering episode %d -> %s", ep, out)
        n = _render_episode(
            src=src,
            episode_index=ep,
            plateau_result=result,
            existing_as=existing_as,
            ep_idx_all=ep_idx,
            fps=fps,
            out_path=out,
            upscale=args.upscale,
            y_max=args.y_max,
            joint_names=joint_names or None,
            drop_plateau=args.drop_plateau,
        )
        logging.info("  wrote %d frames", n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
