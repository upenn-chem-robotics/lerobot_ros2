#!/usr/bin/env python
"""Add an ``action_source`` column to a LeRobot dataset.

Most useful for **base teleop datasets** that were recorded *before* DAgger
existed: every frame in such a dataset is a human teleop frame, so this script
adds ``action_source = 1`` for all rows. The resulting dataset is then
schema-compatible with DAgger datasets (which use ``action_source = 0`` for
policy-prefix frames and ``= 1`` for human-takeover frames) and can be merged
with them via ``lerobot-edit-dataset --operation.type merge`` and trained with
:mod:`lerobot_ros2.cli.train_dagger`.

The script never modifies the source dataset; it writes a new dataset to
``--dst-root``.

Examples::

    # Default: tag every frame as human (action_source=1)
    lerobot-ros-add-action-source \
        --src-root  /lerobot-ros/data/septum_white_merged_base_data_20260510_ds5 \
        --dst-root  /lerobot-ros/data/septum_white_merged_base_data_20260510_ds5_with_as

    # Explicit value (e.g. tag everything as policy-prefix)
    lerobot-ros-add-action-source \
        --src-root  /lerobot-ros/data/some_rollout \
        --dst-root  /lerobot-ros/data/some_rollout_with_as \
        --value     0

    # Custom repo ids (default: derive from folder name with ``local/`` prefix)
    lerobot-ros-add-action-source \
        --src-root  /lerobot-ros/data/my_dataset \
        --dst-root  /lerobot-ros/data/my_dataset_with_as \
        --src-repo-id local/my_dataset \
        --dst-repo-id local/my_dataset_with_as
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _default_repo_id(root: Path) -> str:
    """Derive ``local/<folder_name>`` from a dataset root directory."""
    return f"local/{root.name}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src-root", required=True, type=Path,
        help="Path to the source LeRobot dataset directory.",
    )
    parser.add_argument(
        "--dst-root", required=True, type=Path,
        help="Path where the new dataset (with action_source) will be written. "
             "Must not already exist.",
    )
    parser.add_argument(
        "--src-repo-id", type=str, default=None,
        help="repo_id of the source dataset. Defaults to 'local/<src_root_name>'.",
    )
    parser.add_argument(
        "--dst-repo-id", type=str, default=None,
        help="repo_id for the new dataset. Defaults to 'local/<dst_root_name>'.",
    )
    parser.add_argument(
        "--value", type=int, default=1, choices=(0, 1),
        help="action_source value to write for every frame. "
             "1 = human teleop (default, correct for base recordings); "
             "0 = policy prefix.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    src_root: Path = args.src_root.resolve()
    dst_root: Path = args.dst_root.resolve()
    src_repo_id = args.src_repo_id or _default_repo_id(src_root)
    dst_repo_id = args.dst_repo_id or _default_repo_id(dst_root)

    if not src_root.is_dir():
        print(f"error: --src-root does not exist or is not a directory: {src_root}",
              file=sys.stderr)
        return 2
    if dst_root.exists():
        print(f"error: --dst-root already exists, refuse to overwrite: {dst_root}",
              file=sys.stderr)
        return 2

    ds = LeRobotDataset(repo_id=src_repo_id, root=src_root)
    if "action_source" in ds.meta.features:
        print(f"error: source dataset already has an 'action_source' feature: {src_root}",
              file=sys.stderr)
        return 2

    n = ds.meta.total_frames
    print(f"Adding action_source={args.value} to {n} frames")
    print(f"  src: {src_root}  ({src_repo_id})")
    print(f"  dst: {dst_root}  ({dst_repo_id})")

    values = np.full(n, args.value, dtype=np.int64)
    features = {
        "action_source": (
            values,
            {"dtype": "int64", "shape": (1,), "names": ["action_source"]},
        ),
    }
    add_features(ds, features=features, output_dir=dst_root, repo_id=dst_repo_id)
    print(f"Done. New dataset at {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
