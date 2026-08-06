#!/usr/bin/env python3
"""Compact sparse episode indices in a LeRobot v3.0 dataset.

A recording session that is killed mid-run leaves ``meta/info.json`` ahead of
the data that survived. Both ``total_episodes`` and ``total_frames`` are
allocators rather than descriptions -- LeRobot bumps them as each episode is
handed to the writer, but the parquet footers only land on ``finalize()`` -- so
the next session resumes past the episodes that were lost and the skipped
indices stay empty forever. A dataset with 36 real episodes ends up numbered
``0,1,11..44``.

LeRobot cannot read a dataset in that state:

* ``DatasetReader.try_load`` asks for ``range(total_episodes)`` and bails when
  any index is missing, falling through to a Hub download that fails for a
  local-only dataset.
* ``LeRobotDatasetMetadata.get_video_file_path`` looks episodes up
  *positionally* in ``meta/episodes``, so every index at or past the episode
  count raises ``IndexError`` and the ones below it silently resolve to a
  different episode's row.

This CLI rewrites ``--src`` into ``--dst`` with ``episode_index`` compacted to
``0..N-1`` in ascending order and the global ``index`` column made contiguous
again, then points ``info.json``'s counters at the result. Rows keep their
original file partitioning and order and ``frame_index`` is already per-episode,
so nothing is re-ordered. The packed videos are copied byte-for-byte: episodes
address video by ``from_timestamp``/``to_timestamp``, which this never touches,
so there is no re-encode and no pixel changes.

Unlike ``canonicalize``, this makes no schema changes -- camera keys, joint
count, task vocabulary, extra columns like ``subtask_index`` and the video
codec all survive untouched. Use it when a dataset is fine except for the
numbering.

Examples::

    lerobot-ros-renumber-episodes \
        --src /lerobot-ros/data/smrithi/longhorizon3 \
        --dst /lerobot-ros/data/smrithi/longhorizon3_renumbered
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Copied verbatim into the destination. Everything else under ``meta/`` is
# rewritten, and derived directories (``exports/``, ``images/``) are skipped
# because renumbering makes them stale.
_META_PASSTHROUGH = ("stats.json", "tasks.parquet", "recording_stats.json")
_ROOT_PASSTHROUGH = (
    "experiment_config.yaml",
    "recording_config.yaml",
    "plateau_filter_config.yaml",
)


def _read_data_index(src: Path) -> tuple[pd.DataFrame, dict[str, pa.Table]]:
    """Read every ``data/`` parquet, tracking which row came from where."""
    tables: dict[str, pa.Table] = {}
    frames: list[pd.DataFrame] = []
    for path in sorted((src / "data").rglob("*.parquet")):
        rel = str(path.relative_to(src))
        table = pq.read_table(path)
        tables[rel] = table
        df = table.select(["episode_index", "frame_index", "index"]).to_pandas()
        df["__rel"] = rel
        df["__row"] = range(table.num_rows)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no data parquets under {src / 'data'}")
    return pd.concat(frames, ignore_index=True), tables


def _check_row_order(combined: pd.DataFrame) -> None:
    """Refuse to renumber unless rows are already grouped by ascending episode.

    The global ``index`` column has to stay contiguous within an episode,
    because ``meta/episodes`` addresses each episode as the half-open range
    ``[dataset_from_index, dataset_to_index)``. That holds only if concatenating
    the parquets in file order already yields whole episodes back to back, which
    is how the recorder writes them. Reordering rows to force it would move
    frames out from under the video timestamps, so bail out instead.
    """
    if not combined["episode_index"].is_monotonic_increasing:
        raise SystemExit(
            "rows are not grouped by ascending episode_index across the data "
            "parquets, so renumbering would break the episode ranges in "
            "meta/episodes; refusing to touch this dataset"
        )
    for episode, group in combined.groupby("episode_index", sort=False):
        if list(group["frame_index"]) != list(range(len(group))):
            raise SystemExit(
                f"episode {int(episode)} has a frame_index column that is not "
                f"0..{len(group) - 1}; refusing to renumber"
            )


def _write_data(
    dst: Path, combined: pd.DataFrame, tables: dict[str, pa.Table],
) -> None:
    """Write the data parquets with remapped ``episode_index`` and ``index``."""
    for rel, table in tables.items():
        group = combined.loc[combined["__rel"] == rel].sort_values("__row")
        for column, values in (
            ("episode_index", group["new_episode_index"]),
            ("index", group["new_index"]),
        ):
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                pa.array(values.tolist(), type=table.schema.field(column).type),
            )
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)


def _write_episode_meta(
    src: Path, dst: Path, ep_map: dict[int, int], summary: dict[int, dict[str, int]],
) -> None:
    """Remap ``meta/episodes``, leaving the video timestamps alone."""
    parts = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not parts:
        raise SystemExit(f"no episode metadata under {src / 'meta' / 'episodes'}")
    for path in parts:
        table = pq.read_table(path)
        new_eps = [ep_map[int(v)] for v in table.column("episode_index").to_pylist()]
        for column, values in (
            ("episode_index", new_eps),
            ("dataset_from_index", [summary[e]["dataset_from_index"] for e in new_eps]),
            ("dataset_to_index", [summary[e]["dataset_to_index"] for e in new_eps]),
        ):
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                pa.array(values, type=table.schema.field(column).type),
            )
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)


def _copy_passthrough(src: Path, dst: Path) -> None:
    shutil.copytree(src / "videos", dst / "videos")
    for name in _META_PASSTHROUGH:
        path = src / "meta" / name
        if path.exists():
            shutil.copy2(path, dst / "meta" / name)
    for name in _ROOT_PASSTHROUGH:
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", required=True, type=Path, help="Dataset to read (never modified).")
    p.add_argument("--dst", required=True, type=Path, help="Dataset to write.")
    p.add_argument("--overwrite", action="store_true", help="Replace --dst if it exists.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"not a LeRobot dataset: {src}")
    if src == dst:
        raise SystemExit("--src and --dst must differ; this CLI never rewrites in place")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"destination exists (use --overwrite): {dst}")
        shutil.rmtree(dst)

    combined, tables = _read_data_index(src)
    _check_row_order(combined)

    ep_ids = sorted(int(e) for e in combined["episode_index"].unique())
    ep_map = {old: new for new, old in enumerate(ep_ids)}
    combined["new_episode_index"] = combined["episode_index"].map(ep_map)
    combined["new_index"] = range(len(combined))

    summary = {
        int(episode): {
            "length": int(len(group)),
            "dataset_from_index": int(group["new_index"].min()),
            "dataset_to_index": int(group["new_index"].max()) + 1,
        }
        for episode, group in combined.groupby("new_episode_index", sort=True)
    }
    total_episodes, total_frames = len(summary), len(combined)

    if ep_ids == list(range(total_episodes)):
        logging.info("Episode indices are already 0..%d; copying unchanged", total_episodes - 1)
    else:
        logging.info(
            "Compacting %d episode(s) numbered %d..%d down to 0..%d",
            total_episodes, ep_ids[0], ep_ids[-1], total_episodes - 1,
        )
        moved = [(old, new) for old, new in ep_map.items() if old != new]
        logging.info("  %d episode(s) change index, first: %s", len(moved), moved[:4])
    logging.info(
        "  frames %d, global index will run 0..%d", total_frames, total_frames - 1
    )

    (dst / "meta").mkdir(parents=True)
    _write_data(dst, combined, tables)
    _write_episode_meta(src, dst, ep_map, summary)

    info = json.loads((src / "meta" / "info.json").read_text())
    stale = (info.get("total_episodes"), info.get("total_frames"))
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    logging.info(
        "  info.json counters %s -> (%d, %d)", stale, total_episodes, total_frames
    )

    _copy_passthrough(src, dst)
    logging.info("Wrote %s", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
