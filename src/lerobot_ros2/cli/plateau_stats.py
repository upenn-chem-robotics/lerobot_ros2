#!/usr/bin/env python3
"""Compute plateau (no-op) frame statistics for a LeRobot dataset.

Read-only: this CLI **never modifies the source dataset**. It loads the
``action`` and ``episode_index`` columns from the dataset parquet(s),
runs :func:`lerobot_ros2.plateau.detect_plateaus` with the given
hyperparameters, and prints:

* dataset-level summary (% frames flagged, speed percentiles)
* per-episode top-K worst offenders (most frames flagged, longest runs)
* a histogram of plateau-run lengths

Use this CLI to **tune ``--tau``, ``--min-run``, ``--margin``** before
committing to a new dataset. Pair it with ``lerobot-ros-plateau-visualize``
to visually verify the flagged frames really are operator hesitations
and not legitimate slow motion (e.g. the pour itself).

Examples::

    # Default thresholds (good starting point at 10 Hz datasets)
    lerobot-ros-plateau-stats \
        --src /lerobot-ros/data/pour_20260514_merged_ds5

    # Tighter detection: only mark frames in plateaus of >= 0.8 s
    lerobot-ros-plateau-stats \
        --src /lerobot-ros/data/pour_20260514_merged_ds5 \
        --tau 0.003 --min-run 8 --margin 2

    # Dump full per-episode stats to JSON for further analysis
    lerobot-ros-plateau-stats \
        --src /lerobot-ros/data/pour_20260514_merged_ds5 \
        --json-out /tmp/plateau_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot_ros2.plateau import (
    add_plateau_args,
    detect_plateaus,
    load_action_min_max_from_stats,
    params_from_args,
)


def _load_action_and_episode(src: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``action`` and ``episode_index`` from every parquet under ``src/data``."""
    data_dir = src / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{src} has no data/ directory")
    parts = sorted(data_dir.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet shards under {data_dir}")
    actions: list[np.ndarray] = []
    eps: list[np.ndarray] = []
    for p in parts:
        t = pq.read_table(p, columns=["action", "episode_index"])
        a = np.stack(t.column("action").to_pylist()).astype(np.float64)
        e = np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64)
        actions.append(a)
        eps.append(e)
    return np.concatenate(actions, axis=0), np.concatenate(eps, axis=0)


def _print_summary(result, fps: int, top_k: int) -> None:
    s = result.stats
    p = s["params"]
    total = s["n_total_frames"]
    dropped = s["n_dropped_frames"]
    low = s["n_low_frames"]
    print()
    print("=" * 72)
    print(" Plateau detection summary")
    print("=" * 72)
    print(f"  params: tau={p['tau']}  min_run={p['min_run']}  margin={p['margin']}  norm={p['norm']}")
    print(f"  joint_indices: {p['joint_indices']}  (None = all 7 joints)")
    print(f"  fps={fps}  ->  min_run={p['min_run']} frames = {p['min_run']/fps:.2f} s")
    print()
    print(f"  total frames:   {total}")
    print(f"  low-speed:      {low}  ({100*low/total:.2f}%)")
    print(f"  flagged plateau:{dropped}  ({100*dropped/total:.2f}%)")
    print(f"  episodes:       {s['n_episodes']}")
    print()
    print("  speed (normalized Linf) percentiles:")
    print(f"    p50={s['speed_p50']:.5f}  p90={s['speed_p90']:.5f}  "
          f"p99={s['speed_p99']:.5f}  max={s['speed_max']:.5f}")
    print()

    per_ep = s["per_episode"]
    if not per_ep:
        return

    print(f"  Top {top_k} episodes by % flagged frames:")
    ranked = sorted(per_ep, key=lambda d: d["n_dropped_frames"] / max(d["length"], 1), reverse=True)
    print(f"    {'ep':>4}  {'len':>4}  {'low':>4}  {'runs':>4}  {'drop':>4}  {'maxrun':>6}  {'pct':>5}")
    for row in ranked[:top_k]:
        pct = 100 * row["n_dropped_frames"] / max(row["length"], 1)
        print(f"    {row['episode_index']:>4}  {row['length']:>4}  "
              f"{row['n_low_frames']:>4}  {row['n_plateau_runs']:>4}  "
              f"{row['n_dropped_frames']:>4}  {row['max_run_length']:>6}  "
              f"{pct:>4.1f}%")

    print()
    print(f"  Bottom {min(top_k, len(per_ep))} episodes by % flagged "
          f"(possible smooth demos -- sanity check):")
    for row in ranked[-top_k:][::-1]:
        pct = 100 * row["n_dropped_frames"] / max(row["length"], 1)
        print(f"    {row['episode_index']:>4}  {row['length']:>4}  "
              f"{row['n_low_frames']:>4}  {row['n_plateau_runs']:>4}  "
              f"{row['n_dropped_frames']:>4}  {row['max_run_length']:>6}  "
              f"{pct:>4.1f}%")
    print()


def _print_run_length_histogram(result, fps: int) -> None:
    """Histogram of *plateau-run* lengths across all episodes."""
    bins = Counter()
    for row in result.stats["per_episode"]:
        if row["max_run_length"] > 0:
            bins[row["max_run_length"]] += 1
    if not bins:
        print("  (no plateaus detected -- thresholds may be too strict)")
        return
    print("  Histogram of longest-plateau-per-episode (frames -> count of episodes):")
    for k in sorted(bins.keys()):
        bar = "#" * min(40, bins[k])
        print(f"    {k:>3}f ({k/fps:>4.1f}s)  {bins[k]:>3}  {bar}")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="LeRobot dataset root.")
    add_plateau_args(p)
    p.add_argument("--top-k", type=int, default=10,
                   help="How many top/bottom episodes to print (default: 10).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional path to dump the full stats dict as JSON.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"error: --src does not exist or is not a directory: {src}", file=sys.stderr)
        return 2

    info_path = src / "meta" / "info.json"
    fps = 10
    if info_path.is_file():
        try:
            fps = int(json.loads(info_path.read_text()).get("fps", 10))
        except (OSError, json.JSONDecodeError):
            pass

    stats_json = src / "meta" / "stats.json"
    if not stats_json.is_file():
        print(f"error: missing {stats_json}; cannot normalize action joints. "
              "Run `lerobot-edit-dataset` to recompute stats first.", file=sys.stderr)
        return 2
    a_min, a_max = load_action_min_max_from_stats(stats_json)

    logging.info("Loading action / episode_index from %s ...", src / "data")
    action, ep_idx = _load_action_and_episode(src)
    logging.info("  loaded %d frames across %d episodes", action.shape[0], int(np.unique(ep_idx).size))

    params = params_from_args(args)
    result = detect_plateaus(action, ep_idx, a_min, a_max, params)

    _print_summary(result, fps, top_k=int(args.top_k))
    _print_run_length_histogram(result, fps)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.stats, indent=2))
        print(f"Wrote stats JSON -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
