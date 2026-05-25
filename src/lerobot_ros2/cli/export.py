#!/usr/bin/env python3
"""Export dataset-wide grid media for camera views and timelines.

For every camera in the dataset, this script writes one MP4 grid video per
chunk of episodes. It also writes PNG contact sheets for action and state
timelines, using the same chunking rule.

Each grid is limited to ``--max-items-per-grid`` items. If the dataset has
more items than that, additional files are generated automatically.

Example:

    lerobot-ros-export --dataset-dir data/rama/pour
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lerobot_ros2.data_loader import ArmSpec, TeleopDataset
from lerobot_ros2.visualizer import tile_frames_grid
from lerobot_ros2.video_exporter import (
    build_episode_timeline_figure,
    compute_grid_shape,
    export_episode_grid_video,
)


def _chunked(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def _camera_slug(camera_key: str) -> str:
    return camera_key.replace("/", "_").replace(".", "_")


def _figure_to_rgb_array(fig) -> np.ndarray:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return buffer[..., :3].copy()


def _export_camera_grids(
    ds: TeleopDataset,
    episode_ids: list[int],
    output_dir: Path,
    max_items_per_grid: int,
) -> list[Path]:
    written: list[Path] = []
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for camera_key in ds.camera_keys:
        camera_dir = video_dir / _camera_slug(camera_key)
        camera_dir.mkdir(parents=True, exist_ok=True)
        for chunk in _chunked(episode_ids, max_items_per_grid):
            start_episode = chunk[0]
            end_episode = chunk[-1]
            out_path = camera_dir / f"episodes_{start_episode:06d}_{end_episode:06d}.mp4"
            export_episode_grid_video(ds, chunk, camera_key, output_path=out_path)
            written.append(out_path)
            print(f"Wrote {out_path}", flush=True)

    return written


def _export_timeline_grids(
    ds: TeleopDataset,
    episode_ids: list[int],
    output_dir: Path,
    max_items_per_grid: int,
) -> list[Path]:
    written: list[Path] = []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    arm_specs = ds.arm_specs or [
        ArmSpec(name="arm_1", joint_names=tuple(ds.joint_names), start=0, stop=len(ds.joint_names))
    ]

    for data_type in ("action", "state"):
        data_dir = image_dir / data_type
        data_dir.mkdir(parents=True, exist_ok=True)
        for chunk in _chunked(episode_ids, max_items_per_grid):
            rows, cols = compute_grid_shape(len(chunk))
            tile_w = 720
            tile_h = 405
            target_w = cols * tile_w
            target_h = rows * tile_h

            tile_images: list[np.ndarray] = []
            labels: list[str] = []
            for episode_id in chunk:
                arr = ds.get_episode_actions(episode_id) if data_type == "action" else ds.get_episode_states(episode_id)
                fig = build_episode_timeline_figure(arr, arm_specs, episode_id, data_type)
                tile_images.append(_figure_to_rgb_array(fig))
                labels.append(f"Episode {episode_id}")
                plt.close(fig)

            grid_rgb = tile_frames_grid(tile_images, labels, target_w, target_h, cols)
            start_episode = chunk[0]
            end_episode = chunk[-1]
            out_path = data_dir / f"{data_type}_timelines_{start_episode:06d}_{end_episode:06d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
            written.append(out_path)
            print(f"Wrote {out_path}", flush=True)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export grid videos and timeline grid images from a dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset root directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write exports into. Defaults to <dataset-dir>/exports/grid_media.",
    )
    parser.add_argument(
        "--max-items-per-grid",
        type=int,
        default=30,
        help="Maximum number of episodes per grid file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "exports" / "grid_media"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_items_per_grid <= 0:
        raise ValueError("--max-items-per-grid must be positive")

    ds = TeleopDataset(dataset_dir, preload_frames=False)
    episode_ids = ds.episode_ids
    if not episode_ids:
        raise RuntimeError(f"No episodes found in {dataset_dir}")

    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Episodes: {len(episode_ids)}", flush=True)
    print(f"Cameras: {len(ds.camera_keys)}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    written: list[Path] = []
    written.extend(_export_camera_grids(ds, episode_ids, output_dir, args.max_items_per_grid))
    written.extend(_export_timeline_grids(ds, episode_ids, output_dir, args.max_items_per_grid))

    print(f"Done. Wrote {len(written)} files.", flush=True)


if __name__ == "__main__":
    main()