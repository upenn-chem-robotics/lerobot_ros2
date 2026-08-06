#!/usr/bin/env python
"""Audit and repair the ``subtask_index`` labels of a recorded dataset.

A long-horizon episode is labelled live by the teleoperator's foot pedal (see
``lerobot-ros-record --pedal-right-action subtask``). A missed press shifts every
later label down by one; an extra press skips a label entirely. Neither is
visible in the video, and re-recording a six-minute episode is expensive — so
this tool lets you fix the labels instead of throwing the episode away.

Two modes:

**Audit** (default) prints the subtask segments of every episode and flags the
ones that look wrong::

    lerobot-ros-fix-subtasks --root data/long_horizon

    ep   0  6 segments  OK
              1/6 stir bar     frames    0- 287  ( 28.8s)
              2/6 funnel on    frames  288- 601  ( 31.4s)
              ...
    ep   3  5 segments  SUSPECT: ends on 5/6; subtask 4/6 has no frames

**Repair** rewrites one episode's labels from the frame or timestamp of each
subtask boundary, which you read off the episode's video::

    lerobot-ros-fix-subtasks --root data/long_horizon --episode 3 \
        --boundaries 28.8,60.2,95.5,121.0,150.4 --dry-run

A boundary is where the *next* subtask starts, so N subtasks need N-1
boundaries. Defaults to seconds; pass ``--units frames`` for exact frame
indices. Drop ``--dry-run`` to write. The touched parquet files are backed up
next to themselves first unless you pass ``--no-backup``.

Only the ``subtask_index`` column and its statistics are ever modified. Frame
counts, videos, and every other column are left byte-identical.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.io_utils import write_stats

from lerobot_ros2.dataset_rewrite import episode_parquets, read_info

COLUMN = "subtask_index"
_STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


class Segment(NamedTuple):
    """A maximal run of consecutive frames sharing one ``subtask_index``."""

    label: int
    start: int  # inclusive, episode-relative frame index
    end: int    # inclusive
    name: str = ""

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


class DatasetLayout(NamedTuple):
    root: Path
    fps: float
    episode_lengths: Dict[int, int]
    data_file_for_episode: Dict[int, Path]
    episodes_meta_paths: List[Path]
    subtask_count: int
    subtask_names: List[str]


# ── Pure label logic (unit-tested) ────────────────────────────────────────

def labels_from_boundaries(
    n_frames: int,
    boundaries: Sequence[int],
    subtask_count: int,
) -> np.ndarray:
    """Expand subtask start frames into a per-frame label array.

    ``boundaries[i]`` is the first frame of subtask ``i + 1``, so frames
    ``[0, boundaries[0])`` are subtask 0 and the tail after the last boundary is
    subtask ``subtask_count - 1``.

    Raises ValueError on anything that would produce an empty or out-of-range
    segment, since a silently truncated relabel is worse than a refusal.
    """
    if subtask_count < 1:
        raise ValueError(f"subtask_count must be >= 1, got {subtask_count}")
    if n_frames < subtask_count:
        raise ValueError(
            f"episode has {n_frames} frames but {subtask_count} subtasks; "
            "at least one frame per subtask is required"
        )
    if len(boundaries) != subtask_count - 1:
        raise ValueError(
            f"{subtask_count} subtasks need exactly {subtask_count - 1} boundaries, "
            f"got {len(boundaries)}"
        )

    previous = 0
    for position, boundary in enumerate(boundaries):
        if boundary <= previous:
            reference = "frame 0" if position == 0 else f"boundary {position}"
            raise ValueError(
                f"boundary {position + 1} is at frame {boundary}, which is not after "
                f"{reference} ({previous}); boundaries must strictly increase and "
                "every subtask must contain at least one frame"
            )
        if boundary >= n_frames:
            raise ValueError(
                f"boundary {position + 1} is at frame {boundary}, past the last frame "
                f"({n_frames - 1})"
            )
        previous = boundary

    labels = np.zeros(n_frames, dtype=np.int64)
    for label, boundary in enumerate(boundaries, start=1):
        labels[boundary:] = label
    return labels


def segments_from_labels(labels: Sequence[int], names: Sequence[str] = ()) -> List[Segment]:
    """Collapse a per-frame label array into consecutive runs."""
    segments: List[Segment] = []
    for position, label in enumerate(labels):
        label = int(label)
        if segments and segments[-1].label == label and segments[-1].end == position - 1:
            segments[-1] = segments[-1]._replace(end=position)
            continue
        name = names[label] if 0 <= label < len(names) else ""
        segments.append(Segment(label=label, start=position, end=position, name=name))
    return segments


def diagnose(segments: Sequence[Segment], subtask_count: int) -> List[str]:
    """Human-readable reasons an episode's labels look wrong; empty means OK.

    Catches the two operator errors that are invisible during recording: a
    missed press (the episode ends early, and one label spans two subtasks) and
    an extra press (a label ends up with no frames at all).
    """
    problems: List[str] = []
    if not segments:
        return ["no frames"]

    present = {segment.label for segment in segments}
    last = segments[-1].label

    # A gap before the end means a label was skipped, i.e. one boundary got two
    # presses. A gap only at the end is explained by the "ends early" diagnosis
    # below, so don't blame an extra press for it as well.
    interior_missing = sorted(set(range(last)) - present)
    if interior_missing:
        plural = "s" if len(interior_missing) > 1 else ""
        verb = "have" if len(interior_missing) > 1 else "has"
        problems.append(
            f"subtask{plural} "
            + ", ".join(f"{label + 1}/{subtask_count}" for label in interior_missing)
            + f" {verb} no frames (extra pedal press?)"
        )

    if last != subtask_count - 1:
        problems.append(
            f"ends on {last + 1}/{subtask_count} (missed pedal press?)"
        )

    if segments[0].label != 0:
        problems.append(f"starts on {segments[0].label + 1}/{subtask_count}, not 1")

    labels = [segment.label for segment in segments]
    if labels != sorted(labels):
        problems.append("labels are not monotonically increasing")
    if len(labels) != len(set(labels)):
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        problems.append(
            "label(s) "
            + ", ".join(f"{label + 1}/{subtask_count}" for label in repeated)
            + " appear in more than one run (stepped back mid-episode?)"
        )

    out_of_range = sorted(label for label in present if not 0 <= label < subtask_count)
    if out_of_range:
        problems.append(f"label(s) {out_of_range} outside 0..{subtask_count - 1}")

    return problems


def seconds_to_frames(values: Sequence[float], fps: float) -> List[int]:
    return [int(round(float(value) * fps)) for value in values]


# ── Dataset access ────────────────────────────────────────────────────────

def load_layout(root: Path) -> DatasetLayout:
    """Read the metadata needed to locate and interpret each episode's frames."""
    info = read_info(root)

    if COLUMN not in (info.get("features") or {}):
        raise SystemExit(
            f"{root} has no '{COLUMN}' feature, so there are no subtask labels to "
            "audit or repair. Was it recorded with "
            "--pedal-right-action subtask?"
        )

    fps = float(info.get("fps") or 0.0)
    if fps <= 0:
        raise SystemExit(f"{root}: meta/info.json has no usable fps")

    data_path_template = info.get("data_path") or (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )

    episodes_meta_paths = episode_parquets(root)
    if not episodes_meta_paths:
        raise SystemExit(f"{root}: no meta/episodes/**.parquet found")

    episode_lengths: Dict[int, int] = {}
    data_file_for_episode: Dict[int, Path] = {}
    for meta_path in episodes_meta_paths:
        table = pq.read_table(
            meta_path,
            columns=["episode_index", "length", "data/chunk_index", "data/file_index"],
        )
        frame = table.to_pandas()
        for row in frame.itertuples(index=False):
            episode = int(row.episode_index)
            episode_lengths[episode] = int(row.length)
            data_file_for_episode[episode] = root / data_path_template.format(
                chunk_index=int(getattr(row, "_2")),
                file_index=int(getattr(row, "_3")),
            )

    subtask_count, subtask_names = _read_subtask_config(root)
    return DatasetLayout(
        root=root,
        fps=fps,
        episode_lengths=episode_lengths,
        data_file_for_episode=data_file_for_episode,
        episodes_meta_paths=episodes_meta_paths,
        subtask_count=subtask_count,
        subtask_names=subtask_names,
    )


def _read_subtask_config(root: Path) -> Tuple[int, List[str]]:
    """Subtask count and names as recorded in ``recording_config.yaml``.

    Returns ``(0, [])`` when absent; callers then fall back to what the data
    itself shows.
    """
    config_path = root / "recording_config.yaml"
    if not config_path.is_file():
        return 0, []
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError:
        return 0, []
    recording = cfg.get("recording") or {}
    count = int(recording.get("subtask_count") or 0)
    names = [str(name or "") for name in (recording.get("subtask_names") or [])]
    return count, names


def read_episode_labels(layout: DatasetLayout, episode: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(row_positions, labels)`` for one episode, ordered by frame_index.

    ``row_positions`` indexes into the episode's data parquet, so a caller can
    scatter new labels back without assuming episodes are stored contiguously.
    """
    data_path = layout.data_file_for_episode.get(episode)
    if data_path is None:
        raise SystemExit(f"episode {episode} is not in this dataset")
    if not data_path.is_file():
        raise SystemExit(f"episode {episode}: data file missing: {data_path}")

    table = pq.read_table(data_path, columns=["episode_index", "frame_index", COLUMN])
    episode_column = table.column("episode_index").to_numpy(zero_copy_only=False)
    positions = np.flatnonzero(episode_column.astype(np.int64) == episode)
    if positions.size == 0:
        raise SystemExit(f"episode {episode}: no rows found in {data_path}")

    frame_index = table.column("frame_index").to_numpy(zero_copy_only=False)[positions]
    order = np.argsort(frame_index, kind="stable")
    positions = positions[order]
    labels = table.column(COLUMN).to_numpy(zero_copy_only=False)[positions]
    return positions, np.asarray([_as_int(v) for v in labels], dtype=np.int64)


def _as_int(value: object) -> int:
    """Read a label that may be stored as a scalar or a shape-(1,) array."""
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        raise ValueError("empty subtask_index entry")
    return int(array[0])


def effective_subtask_count(layout: DatasetLayout, labels: Sequence[int]) -> int:
    if layout.subtask_count > 0:
        return layout.subtask_count
    return int(max(labels)) + 1 if len(labels) else 0


# ── Audit ─────────────────────────────────────────────────────────────────

def audit(layout: DatasetLayout, episodes: Optional[Sequence[int]] = None) -> int:
    """Print each episode's segments; return the number of suspect episodes."""
    targets = sorted(layout.episode_lengths) if episodes is None else sorted(episodes)
    suspect = 0

    if layout.subtask_count:
        print(
            f"{layout.root}: {len(targets)} episode(s), {layout.subtask_count} subtasks "
            f"@ {layout.fps:g} fps"
        )
    else:
        print(
            f"{layout.root}: {len(targets)} episode(s) @ {layout.fps:g} fps "
            "(no subtask_count in recording_config.yaml; inferring from the data)"
        )

    for episode in targets:
        _, labels = read_episode_labels(layout, episode)
        count = effective_subtask_count(layout, labels)
        segments = segments_from_labels(labels, layout.subtask_names)
        problems = diagnose(segments, count)
        if problems:
            suspect += 1
        status = "OK" if not problems else "SUSPECT: " + "; ".join(problems)
        print(f"\nep {episode:>3}  {len(segments)} segment(s)  {status}")
        for segment in segments:
            name = f" {segment.name}" if segment.name else ""
            print(
                f"          {segment.label + 1}/{count}{name:<14} "
                f"frames {segment.start:>5}-{segment.end:<5} "
                f"({segment.n_frames / layout.fps:6.1f}s)"
            )

    print(
        f"\n{len(targets) - suspect}/{len(targets)} episode(s) look OK, {suspect} suspect."
    )
    if suspect:
        print(
            "Repair one with:\n"
            "  lerobot-ros-fix-subtasks --root <root> --episode <n> "
            "--boundaries <s1,s2,...> --dry-run"
        )
    return suspect


# ── Repair ────────────────────────────────────────────────────────────────

def relabel_episode(
    layout: DatasetLayout,
    episode: int,
    boundary_frames: Sequence[int],
    dry_run: bool = True,
    backup: bool = True,
) -> None:
    positions, old_labels = read_episode_labels(layout, episode)
    n_frames = len(positions)
    expected = layout.episode_lengths.get(episode)
    if expected is not None and expected != n_frames:
        raise SystemExit(
            f"episode {episode}: meta says {expected} frames but the data file has "
            f"{n_frames}; refusing to touch an inconsistent dataset"
        )

    count = effective_subtask_count(layout, old_labels)
    new_labels = labels_from_boundaries(n_frames, boundary_frames, count)

    print(f"episode {episode}: {n_frames} frames @ {layout.fps:g} fps, {count} subtasks")
    print("\nbefore:")
    _print_segments(segments_from_labels(old_labels, layout.subtask_names), count, layout.fps)
    print("\nafter:")
    _print_segments(segments_from_labels(new_labels, layout.subtask_names), count, layout.fps)

    changed = int(np.count_nonzero(old_labels != new_labels))
    print(f"\n{changed} of {n_frames} frame label(s) would change")
    if changed == 0:
        print("Nothing to do.")
        return
    if dry_run:
        print("Dry run: nothing written. Re-run without --dry-run to apply.")
        return

    data_path = layout.data_file_for_episode[episode]
    _write_labels(data_path, positions, new_labels, backup=backup)
    print(f"wrote {data_path}")

    _refresh_episode_stats(layout, episode, new_labels, backup=backup)
    _refresh_global_stats(layout)
    print("Done.")


def _print_segments(segments: Sequence[Segment], count: int, fps: float) -> None:
    for segment in segments:
        name = f" {segment.name}" if segment.name else ""
        print(
            f"  {segment.label + 1}/{count}{name:<14} frames "
            f"{segment.start:>5}-{segment.end:<5} ({segment.n_frames / fps:6.1f}s)"
        )


def _backup(path: Path) -> Path:
    destination = path.with_suffix(path.suffix + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, destination)
    print(f"backup: {destination}")
    return destination


def _write_labels(
    path: Path,
    positions: np.ndarray,
    new_labels: np.ndarray,
    backup: bool = True,
) -> None:
    """Replace ``subtask_index`` at ``positions`` and rewrite the parquet.

    Only that one column is rebuilt; every other column is handed back to
    pyarrow untouched, so other episodes sharing this file are unaffected.
    """
    table = pq.read_table(path)
    field_index = table.schema.get_field_index(COLUMN)
    if field_index < 0:
        raise SystemExit(f"{path}: no '{COLUMN}' column")
    field = table.schema.field(field_index)

    current = table.column(COLUMN).to_numpy(zero_copy_only=False)
    is_nested = pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
    flat = np.asarray([_as_int(v) for v in current], dtype=np.int64)
    flat[positions] = new_labels

    if is_nested:
        replacement = pa.array([[int(v)] for v in flat], type=field.type)
    else:
        replacement = pa.array(flat, type=field.type)

    if backup:
        _backup(path)
    pq.write_table(table.set_column(field_index, field, replacement), path)


def _refresh_episode_stats(
    layout: DatasetLayout,
    episode: int,
    new_labels: np.ndarray,
    backup: bool = True,
) -> None:
    """Recompute this episode's ``stats/subtask_index/*`` row in meta/episodes."""
    stats = get_feature_stats(new_labels.reshape(-1, 1).astype(np.float64), axis=0, keepdims=True)
    for meta_path in layout.episodes_meta_paths:
        table = pq.read_table(meta_path)
        columns = [name for name in table.schema.names if name.startswith(f"stats/{COLUMN}/")]
        if not columns:
            continue
        episode_column = table.column("episode_index").to_numpy(zero_copy_only=False)
        rows = np.flatnonzero(episode_column.astype(np.int64) == episode)
        if rows.size == 0:
            continue

        updated = table
        for column in columns:
            stat_name = column.rsplit("/", 1)[-1]
            if stat_name not in stats:
                continue
            field_index = updated.schema.get_field_index(column)
            field = updated.schema.field(field_index)
            values = updated.column(column).to_pylist()
            value = np.asarray(stats[stat_name]).reshape(-1).tolist()
            for row in rows:
                values[int(row)] = value
            updated = updated.set_column(
                field_index, field, pa.array(values, type=field.type)
            )

        if backup:
            _backup(meta_path)
        pq.write_table(updated, meta_path)
        print(f"wrote {meta_path} (episode {episode} {COLUMN} stats)")
        return


def _refresh_global_stats(layout: DatasetLayout) -> None:
    """Recompute the dataset-wide ``subtask_index`` entry in meta/stats.json.

    Computed straight from every label in the dataset rather than aggregated
    from per-episode stats, which for one integer column is both simpler and
    exact.
    """
    stats_path = layout.root / "meta" / "stats.json"
    if not stats_path.is_file():
        return

    all_labels: List[int] = []
    for data_path in sorted(set(layout.data_file_for_episode.values())):
        if not data_path.is_file():
            continue
        column = pq.read_table(data_path, columns=[COLUMN]).column(COLUMN)
        all_labels.extend(_as_int(value) for value in column.to_pylist())
    if not all_labels:
        return

    array = np.asarray(all_labels, dtype=np.float64).reshape(-1, 1)
    stats = get_feature_stats(array, axis=0, keepdims=True)

    # min/max/count stay integral so the entry looks exactly like the one
    # LeRobot wrote for this column when the dataset was created.
    integral = {"min", "max", "count"}
    existing = json.loads(stats_path.read_text())
    existing[COLUMN] = {
        name: np.asarray(value).reshape(-1).astype(
            np.int64 if name in integral else np.float64
        )
        for name, value in stats.items()
        if name in _STAT_NAMES
    }
    write_stats({key: _to_arrays(value) for key, value in existing.items()}, layout.root)
    print(f"wrote {stats_path} ({COLUMN} stats over {len(all_labels)} frames)")


def _to_arrays(value: object) -> object:
    if isinstance(value, dict):
        return {key: _to_arrays(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", required=True, type=Path,
        help="Dataset root directory, e.g. data/long_horizon.",
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode to repair. Omit to audit every episode instead.",
    )
    parser.add_argument(
        "--boundaries", type=str, default=None,
        help="Comma-separated start of each subtask after the first, in --units. "
             "N subtasks need N-1 values, e.g. '28.8,60.2,95.5,121.0,150.4' for 6.",
    )
    parser.add_argument(
        "--units", choices=("seconds", "frames"), default="seconds",
        help="Unit of --boundaries (default: %(default)s). Seconds are what a "
             "video player shows; frames are exact.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the before/after segments without writing anything.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip the timestamped .bak copy of each parquet file that is rewritten.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    root: Path = args.root.resolve()
    layout = load_layout(root)

    if args.episode is None:
        if args.boundaries:
            print("error: --boundaries needs --episode", file=sys.stderr)
            return 2
        return 1 if audit(layout) else 0

    if not args.boundaries:
        # An --episode with no boundaries is a request to inspect just that one.
        return 1 if audit(layout, episodes=[args.episode]) else 0

    try:
        raw = [value for value in args.boundaries.split(",") if value.strip()]
        values = [float(value) for value in raw]
    except ValueError:
        print(f"error: could not parse --boundaries {args.boundaries!r}", file=sys.stderr)
        return 2

    if args.units == "seconds":
        boundary_frames = seconds_to_frames(values, layout.fps)
        print(
            f"boundaries {values} s -> frames {boundary_frames} at {layout.fps:g} fps"
        )
    else:
        boundary_frames = [int(round(value)) for value in values]

    try:
        relabel_episode(
            layout,
            args.episode,
            boundary_frames,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
