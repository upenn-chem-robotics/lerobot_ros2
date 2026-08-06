#!/usr/bin/env python3
"""Shared plumbing for the CLIs that rewrite a LeRobot v3.0 dataset.

``downsample``, ``canonicalize``, ``trim_tail``, ``fix_subtasks`` and
``renumber_episodes`` all read a source dataset, transform it, and write a new
one. Their *transformations* are genuinely different and stay in their own
modules; what they share -- locating ``meta/``, reading and writing the JSON
sidecars with a consistent shape, and building a row index over ``data/`` --
lives here so a schema change lands in one place.

LeRobot's own ``lerobot.datasets.io_utils`` also exposes ``write_info`` /
``write_stats``; ``trim_tail`` uses those because it already depends on
LeRobot's stats aggregation. The writers here exist so the pure-metadata CLIs
can be imported and tested without LeRobot installed, and they produce
byte-identical output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# LeRobot writes its JSON sidecars with this indent; matching it keeps rewritten
# datasets diffable against freshly recorded ones.
JSON_INDENT = 4


# ── Locating things ──────────────────────────────────────────────────────

def info_path(root: Path) -> Path:
    return root / "meta" / "info.json"


def stats_path(root: Path) -> Path:
    return root / "meta" / "stats.json"


def is_dataset(root: Path) -> bool:
    return info_path(root).is_file()


def require_dataset(root: Path) -> None:
    """Exit unless ``root`` looks like a LeRobot dataset."""
    if not is_dataset(root):
        raise SystemExit(f"not a LeRobot dataset (no meta/info.json): {root}")


def data_parquets(root: Path) -> list[Path]:
    """Every ``data/`` parquet, in stable path order."""
    return sorted((root / "data").rglob("*.parquet"))


def episode_parquets(root: Path) -> list[Path]:
    """Every ``meta/episodes/`` parquet, in stable path order."""
    return sorted((root / "meta" / "episodes").rglob("*.parquet"))


_CHUNK_RE = re.compile(r"chunk-(\d+)")
_FILE_RE = re.compile(r"file-(\d+)")


def chunk_file_indices(path: Path) -> tuple[int, int]:
    """Pull ``(chunk_index, file_index)`` out of a ``chunk-NNN/file-NNN`` path."""
    chunk = _CHUNK_RE.search(path.as_posix())
    file_ = _FILE_RE.search(path.as_posix())
    if chunk is None or file_ is None:
        raise SystemExit(f"cannot parse chunk/file index from {path}")
    return int(chunk.group(1)), int(file_.group(1))


# ── meta/info.json ───────────────────────────────────────────────────────

def read_info(root: Path) -> dict:
    """Read ``meta/info.json``, exiting if ``root`` is not a dataset."""
    require_dataset(root)
    return json.loads(info_path(root).read_text())


def write_info(root: Path, info: dict) -> None:
    out = info_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=JSON_INDENT))


def set_episode_counters(info: dict, total_episodes: int, total_frames: int) -> dict:
    """Update the counters and the ``train`` split to match a rewritten dataset.

    Mutates and returns ``info``. Every rewrite that drops or renumbers
    episodes has to do this, and a stale counter here makes the dataset
    unloadable rather than merely wrong.
    """
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


# ── meta/stats.json ──────────────────────────────────────────────────────

def read_stats(root: Path) -> dict | None:
    """Read ``meta/stats.json``, or ``None`` when the dataset has no stats."""
    path = stats_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_stats(root: Path, stats: dict) -> None:
    out = stats_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=JSON_INDENT))


def replace_count_leaves(value: object, count: int) -> object:
    """Rewrite every numeric leaf of a ``.../count`` stat entry to ``count``."""
    if isinstance(value, list):
        return [replace_count_leaves(item, count) for item in value]
    if isinstance(value, (int, float)):
        return count
    return value


# ── data/ row index ──────────────────────────────────────────────────────

def load_data_index(
    root: Path,
    columns: Sequence[str] = ("episode_index", "index"),
    optional_columns: Iterable[str] = (),
    rel_col: str = "__rel",
    row_col: str = "__row",
    sort_by_index: bool = True,
    allow_empty: bool = False,
) -> tuple[pd.DataFrame, dict[str, pa.Table]]:
    """Build a row index over every ``data/`` parquet.

    Returns ``(frame, tables)`` where ``frame`` has one row per dataset row
    carrying ``columns`` plus ``rel_col``/``row_col`` (the source parquet path
    relative to ``root`` and the row offset within it), and ``tables`` maps
    that same relative path to the loaded :class:`pyarrow.Table`. Together they
    let a caller reorder or filter rows and still write each row back to the
    right file.

    ``optional_columns`` are included only where present in the schema, so
    callers can pick up e.g. ``action_source`` without knowing in advance
    whether the dataset carries it.
    """
    tables: dict[str, pa.Table] = {}
    frames: list[pd.DataFrame] = []

    for path in data_parquets(root):
        rel = str(path.relative_to(root))
        table = pq.read_table(path)
        tables[rel] = table

        selected = list(columns)
        selected += [c for c in optional_columns if c in table.schema.names]
        df = table.select(selected).to_pandas()
        df[rel_col] = rel
        df[row_col] = range(table.num_rows)
        frames.append(df)

    if not frames:
        if not allow_empty:
            raise SystemExit(f"no data parquets under {root / 'data'}")
        empty = pd.DataFrame(columns=[*columns, rel_col, row_col])
        return empty, tables

    combined = pd.concat(frames, ignore_index=True)
    if sort_by_index:
        combined = combined.sort_values("index").reset_index(drop=True)
    return combined, tables
