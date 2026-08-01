#!/usr/bin/env python
"""Emit a new dataset with ``action_source`` reflecting plateau filtering.

This is a plateau-aware sibling of :mod:`lerobot_ros2.cli.add_action_source_to_base`.
Where ``add_action_source_to_base`` tags every frame as human
(``action_source = 1``), this CLI tags as ``0`` (=> not sampled as an
anchor by :class:`ActionSourceAwareEpisodeSampler`) every frame that
falls inside a long enough no-motion run, and keeps everything else as
``1`` (or preserves the existing ``action_source == 0`` from a DAgger
merge).

Combine semantics:

    new_action_source[t] = old_action_source[t] AND NOT plateau_mask[t]

where ``old_action_source`` is loaded from the source dataset if it
exists, or treated as "all 1" otherwise. This means:

* A plateau frame in a base demo (``old=1``) -> ``new=0`` (skipped).
* A DAgger policy-prefix frame (``old=0``) -> ``new=0`` (still skipped).
* A motion frame (``old=1``, not plateau) -> ``new=1`` (still trained on).

The source dataset is **never modified**. A new dataset is written to
``--dst-root`` with the new ``action_source`` column added, and a
small ``plateau_filter_config.yaml`` is written alongside so the exact
hyperparameters used are reproducible.

Examples::

    lerobot-ros-add-action-source-with-plateau \
        --src-root /lerobot-ros/data/pour_20260514_merged_ds5 \
        --dst-root /lerobot-ros/data/pour_20260514_merged_ds5_plateau \
        --tau 0.005 --min-run 5 --margin 2

    # Tighter detection
    lerobot-ros-add-action-source-with-plateau \
        --src-root /lerobot-ros/data/pour_20260514_merged_ds5 \
        --dst-root /lerobot-ros/data/pour_20260514_merged_ds5_plateau_tight \
        --tau 0.003 --min-run 8 --margin 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_ros2.plateau import (
    PlateauParams,
    detect_plateaus,
    load_action_min_max_from_stats,
)


def _default_repo_id(root: Path) -> str:
    return f"local/{root.name}"


def _load_action_episode_existing_as(
    src: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read ``action``, ``episode_index``, and optional ``action_source`` from parquets."""
    data_dir = src / "data"
    parts = sorted(data_dir.rglob("*.parquet"))
    # A chunk holds no rows when every episode it covered was dropped upstream
    # (e.g. by downsample). It contributes no frames, so drop it before the
    # schema probe and the concatenation below.
    parts = [p for p in parts if pq.read_metadata(p).num_rows > 0]
    if not parts:
        raise SystemExit(f"No non-empty parquet files found under {data_dir}")
    actions: list[np.ndarray] = []
    eps: list[np.ndarray] = []
    has_existing = "action_source" in pq.read_schema(parts[0]).names
    sources: list[np.ndarray] = []
    for p in parts:
        cols = ["action", "episode_index"]
        if has_existing:
            cols.append("action_source")
        t = pq.read_table(p, columns=cols)
        actions.append(np.stack(t.column("action").to_pylist()).astype(np.float64))
        eps.append(np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64))
        if has_existing:
            raw = t.column("action_source").to_pylist()
            sources.append(
                np.asarray([v[0] if isinstance(v, (list, tuple)) else v for v in raw], dtype=np.int64)
            )
    action = np.concatenate(actions, axis=0)
    ep_idx = np.concatenate(eps, axis=0)
    existing = np.concatenate(sources, axis=0) if has_existing else None
    return action, ep_idx, existing


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", required=True, type=Path)
    p.add_argument("--dst-root", required=True, type=Path)
    p.add_argument("--src-repo-id", type=str, default=None)
    p.add_argument("--dst-repo-id", type=str, default=None)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--min-run", type=int, default=5)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--norm", choices=("linf", "l2"), default="linf")
    p.add_argument("--joints", type=str, default=None,
                   help="Comma-separated joint indices to consider; default = all.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src_root: Path = args.src_root.resolve()
    dst_root: Path = args.dst_root.resolve()
    src_repo_id = args.src_repo_id or _default_repo_id(src_root)
    dst_repo_id = args.dst_repo_id or _default_repo_id(dst_root)

    if not src_root.is_dir():
        print(f"error: --src-root does not exist or is not a directory: {src_root}", file=sys.stderr)
        return 2
    if dst_root.exists():
        print(f"error: --dst-root already exists, refuse to overwrite: {dst_root}", file=sys.stderr)
        return 2

    stats_json = src_root / "meta" / "stats.json"
    if not stats_json.is_file():
        print(f"error: missing {stats_json}; cannot normalize action joints.", file=sys.stderr)
        return 2

    a_min, a_max = load_action_min_max_from_stats(stats_json)
    action, ep_idx, existing_as = _load_action_episode_existing_as(src_root)

    joints = None
    if args.joints:
        joints = tuple(int(x) for x in args.joints.split(","))

    params = PlateauParams(
        tau=float(args.tau), min_run=int(args.min_run), margin=int(args.margin),
        norm=args.norm, joint_indices=joints,
    )
    result = detect_plateaus(action, ep_idx, a_min, a_max, params)

    if existing_as is None:
        old = np.ones(action.shape[0], dtype=np.int64)
        had_existing = False
    else:
        old = existing_as.astype(np.int64)
        had_existing = True

    new_as = old.copy()
    new_as[result.plateau_mask] = 0

    n = int(new_as.size)
    n_kept = int((new_as == 1).sum())
    n_dropped = int((new_as == 0).sum())
    n_pre_existing_zero = int((old == 0).sum()) if had_existing else 0
    n_plateau_flipped = int((result.plateau_mask & (old == 1)).sum())

    print()
    print("=" * 72)
    print(" add_action_source_with_plateau")
    print("=" * 72)
    print(f"  src:  {src_root}  ({src_repo_id})")
    print(f"  dst:  {dst_root}  ({dst_repo_id})")
    print(f"  had existing action_source column: {had_existing}")
    print(f"  params: tau={params.tau}  min_run={params.min_run}  "
          f"margin={params.margin}  norm={params.norm}  joints={params.joint_indices}")
    print()
    print(f"  total frames:           {n}")
    print(f"  -> action_source = 1:   {n_kept}   ({100*n_kept/n:.2f}%)")
    print(f"  -> action_source = 0:   {n_dropped} ({100*n_dropped/n:.2f}%)")
    print(f"     of which: pre-existing 0:  {n_pre_existing_zero}")
    print(f"               plateau flipped: {n_plateau_flipped}")
    print()

    ds = LeRobotDataset(repo_id=src_repo_id, root=src_root)
    if "action_source" in ds.meta.features:
        print(
            "warning: source dataset already has an 'action_source' feature. "
            "Cannot use add_features() to add it again -- this CLI requires the "
            "source dataset to NOT yet have action_source. Use a different "
            "merge workflow (lerobot-edit-dataset rewrite-column) to overwrite.",
            file=sys.stderr,
        )
        return 2

    features = {
        "action_source": (
            new_as.astype(np.int64),
            {"dtype": "int64", "shape": (1,), "names": ["action_source"]},
        ),
    }
    add_features(ds, features=features, output_dir=dst_root, repo_id=dst_repo_id)

    cfg_path = dst_root / "plateau_filter_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "source_dataset": str(src_root),
                "source_repo_id": src_repo_id,
                "params": result.stats["params"],
                "summary": {
                    "n_total_frames": n,
                    "n_action_source_1": n_kept,
                    "n_action_source_0": n_dropped,
                    "n_pre_existing_zero": n_pre_existing_zero,
                    "n_plateau_flipped": n_plateau_flipped,
                    "speed_p50": result.stats["speed_p50"],
                    "speed_p90": result.stats["speed_p90"],
                    "speed_p99": result.stats["speed_p99"],
                    "speed_max": result.stats["speed_max"],
                },
            },
            sort_keys=False,
        )
    )
    print(f"  wrote {cfg_path}")
    print(f"  Done. New dataset at {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
