#!/usr/bin/env python3
"""Rename image keys and downsample videos in a LeRobot v3.0 dataset.

The image-key rename map is derived automatically from the source dataset
by stripping the ``camera_NN_`` prefix from any ``observation.images.*``
video feature, e.g.:

    observation.images.camera_00_front           -> observation.images.front
    observation.images.camera_01_left_wrist_top  -> observation.images.left_wrist_top
    observation.images.camera_02_perspective     -> observation.images.perspective

This works for any number of cameras (3, 4, 5, ...) and any index numbering
without code changes — whatever video keys the source ``info.json`` lists
are the ones that get renamed and downsampled. To exclude a camera from the
output, use ``--exclude-cameras`` (matched against the *stripped* name,
e.g. ``--exclude-cameras back right_wrist_top``).

To shrink ``observation.state`` / ``action`` from a bimanual recording into
a single-arm subset, use ``--exclude-arms`` (matched against the joint
``names`` listed in ``info.json``). Every joint whose name starts with one
of the given prefixes is removed from both the per-frame data parquets and
the per-feature / per-episode statistics. The matching block under the same
name in ``experiment_config.yaml`` is also stripped from the destination.

Videos are re-encoded at 1/DOWNSAMPLE resolution (default 5x: 720x1280 -> 144x256).

Alternatively, pass ``--out-size H W`` to scale every camera to one common
resolution (required by the diffusion policy, which expects all cameras to
share a shape). Combine with ``--crop NAME TOP LEFT HEIGHT WIDTH`` to crop a
per-camera ROI at full resolution *before* scaling, e.g. to zoom the dataset in
on a small region. The source ROI for every camera is recorded in the output
``info.json`` (``source_crop`` / ``source_shape``) so deploy/dagger can
reproduce the identical crop-then-resize at inference time.

Optionally drop the first N frames from every episode when rewriting the
trajectory and episode metadata. When trimming, per-row timestamps are
rebased to start at 0 for each episode so they stay aligned with the
shifted video metadata.

Usage:
    lerobot-ros-downsample --src /path/to/src_dataset --dst /path/to/dst_dataset \
        [--downsample N] [--out-size H W] [--crop NAME TOP LEFT HEIGHT WIDTH ...] \
        [--skip-first-frames N] [--delete-episodes E1 E2 ...] [--exclude-cameras NAME ...] \
        [--exclude-arms PREFIX ...]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from lerobot_ros2.dataset_rewrite import (
    load_data_index,
    read_info,
    read_stats,
    replace_count_leaves,
    require_dataset,
    write_info,
    write_stats,
)

# Populated at runtime by ``derive_key_map`` from the source dataset.
# Maps original feature keys (``observation.images.camera_NN_<name>``) to the
# stripped names (``observation.images.<name>``) used downstream by
# record/deploy/training configs.
KEY_MAP: dict[str, str] = {}

# Stripped camera names (e.g. ``"back"``) that should be skipped entirely.
EXCLUDED_CAMERAS: set[str] = set()

# Joint-name prefixes (e.g. ``"left"``) whose joints should be removed from
# ``observation.state`` / ``action`` and whose matching top-level block in
# ``experiment_config.yaml`` should be stripped from the destination.
EXCLUDED_ARM_PREFIXES: set[str] = set()

# Per-feature index lists describing which dimensions of
# ``observation.state`` / ``action`` survive after applying
# ``EXCLUDED_ARM_PREFIXES``. Empty when no arm exclusion is requested, so
# every callsite can short-circuit with ``if not ARM_KEEP_INDICES: ...``.
ARM_KEEP_INDICES: dict[str, list[int]] = {}

# Original dimensionality of each affected feature, captured from
# ``info.json`` before slicing. Used to safely guard the per-axis stat
# slicing in ``stats.json`` and in the per-episode stats parquets.
ARM_ORIG_DIMS: dict[str, int] = {}

# Feature keys that may be sliced by ``--exclude-arms``.
STATE_ACTION_KEYS: tuple[str, ...] = ("observation.state", "action")

# Per-camera full-res crop ROIs keyed by *stripped* camera name (e.g. "front").
# Each value is ``{top, left, height, width}`` in source-video pixel coords and
# is applied by ffmpeg before scaling to ``OUT_SIZE``. Cameras absent here are
# scaled full-frame. Populated in ``main`` from ``--crop``.
CROPS: dict[str, dict[str, int]] = {}

# Common output ``(H, W)`` that every camera is scaled to after any crop. When
# ``None`` the legacy 1/downsample whole-frame behaviour is used. Set from
# ``--out-size``.
OUT_SIZE: tuple[int, int] | None = None

_CAMERA_PREFIX_RE = re.compile(r"^observation\.images\.camera_\d+_(?P<name>.+)$")
_EPISODE_STATE_ACTION_COL_RE = re.compile(
    r"^stats/(?P<feature>observation\.state|action)/[^/]+$"
)


def _arm_prefix_matches(name: str, prefixes: set[str]) -> bool:
    """True if ``name`` is exactly one of ``prefixes`` or starts with ``<prefix>_``.

    Matching is case-insensitive. Trailing underscores on the user-supplied
    prefix are ignored, so ``--exclude-arms left`` and ``--exclude-arms left_``
    behave identically.
    """
    if not prefixes:
        return False
    lname = name.lower()
    for raw in prefixes:
        prefix = raw.lower().rstrip("_")
        if not prefix:
            continue
        if lname == prefix or lname.startswith(prefix + "_"):
            return True
    return False


def _stripped_cam_name(feature_key: str) -> str:
    """Return the camera name after the ``observation.images.`` prefix.

    Works for both renamed keys (``observation.images.front``) and any raw key
    that still carries the prefix. Falls back to the whole string otherwise.
    """
    marker = "observation.images."
    idx = feature_key.rfind(marker)
    if idx == -1:
        return feature_key
    return feature_key[idx + len(marker):]


def parse_crops(raw_entries: list[list[str]]) -> dict[str, dict[str, int]]:
    """Parse ``--crop NAME TOP LEFT HEIGHT WIDTH`` entries into a dict.

    Keyed by the stripped camera name; values are ``{top, left, height,
    width}`` ints. Raises ``SystemExit`` on malformed / non-positive specs so a
    typo can't silently produce a bad dataset.
    """
    crops: dict[str, dict[str, int]] = {}
    for entry in raw_entries or []:
        if len(entry) != 5:
            raise SystemExit(
                f"--crop expects 5 values NAME TOP LEFT HEIGHT WIDTH; got {entry}"
            )
        name = str(entry[0]).strip()
        if not name:
            raise SystemExit(f"--crop has an empty NAME in {entry}")
        try:
            top, left, height, width = (int(v) for v in entry[1:])
        except ValueError:
            raise SystemExit(f"--crop TOP LEFT HEIGHT WIDTH must be integers; got {entry}")
        if height <= 0 or width <= 0:
            raise SystemExit(f"--crop {name} height/width must be positive; got {entry}")
        if top < 0 or left < 0:
            raise SystemExit(f"--crop {name} top/left must be >= 0; got {entry}")
        if name in crops:
            raise SystemExit(f"--crop specified more than once for camera {name!r}")
        crops[name] = {"top": top, "left": left, "height": height, "width": width}
    return crops


def validate_crops_against_source(
    source_info: dict,
    key_map: dict[str, str],
    crops: dict[str, dict[str, int]],
) -> None:
    """Ensure every ``--crop`` names a kept camera and fits its source frame.

    ``source_info`` is the parsed source ``info.json``. Bounds are checked
    against each camera's declared ``[height, width, channels]`` shape.
    """
    kept_names = {_stripped_cam_name(new_key) for new_key in key_map.values()}
    features = source_info.get("features") or {}
    for name, spec in crops.items():
        if name not in kept_names:
            raise SystemExit(
                f"--crop names camera {name!r}, which is not a kept camera. "
                f"Available: {sorted(kept_names)}"
            )
        # Find the source feature for this stripped name to read its dims.
        src_key = next(
            (old for old, new in key_map.items() if _stripped_cam_name(new) == name),
            None,
        )
        ft = features.get(src_key) if src_key else None
        shape = (ft or {}).get("shape") if isinstance(ft, dict) else None
        if not shape or len(shape) < 2:
            continue
        h, w = int(shape[0]), int(shape[1])
        if spec["top"] + spec["height"] > h or spec["left"] + spec["width"] > w:
            raise SystemExit(
                f"--crop {name} {spec} exceeds source frame (H={h}, W={w})."
            )


def derive_key_map(src: Path, exclude: set[str] | None = None) -> dict[str, str]:
    """Build the rename map from the source dataset's ``info.json`` features.

    Any video feature key matching ``observation.images.camera_NN_<name>`` is
    mapped to ``observation.images.<name>``. Cameras whose stripped name is
    in ``exclude`` are dropped from the map (and therefore not copied or
    re-encoded into the destination).
    """
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"Cannot derive camera mapping: missing {info_path}")

    info = json.loads(info_path.read_text())
    features = info.get("features") or {}
    exclude = set(exclude or [])

    mapping: dict[str, str] = {}
    excluded_seen: set[str] = set()
    skipped_unprefixed: list[str] = []

    for key, ft in features.items():
        if not isinstance(ft, dict) or ft.get("dtype") != "video":
            continue
        match = _CAMERA_PREFIX_RE.match(key)
        if match is None:
            # Already stripped or unexpected format; pass through unchanged.
            skipped_unprefixed.append(key)
            continue
        stripped = match.group("name")
        if stripped in exclude:
            excluded_seen.add(stripped)
            continue
        mapping[key] = f"observation.images.{stripped}"

    print("[info] derived camera rename map:")
    if not mapping:
        print("         (no camera_NN_ prefixed video keys found)")
    for old, new in mapping.items():
        print(f"         {old}  ->  {new}")

    if skipped_unprefixed:
        print(f"[info] passing through unrenamed video keys: {skipped_unprefixed}")

    if exclude:
        missing = exclude - excluded_seen
        if excluded_seen:
            print(f"[info] excluded cameras (dropped): {sorted(excluded_seen)}")
        if missing:
            print(
                f"[warn] --exclude-cameras requested {sorted(missing)} but no "
                f"matching video features were found in the source dataset"
            )

    return mapping


def derive_arm_indices(
    src: Path,
    exclude_prefixes: set[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Resolve which dims of ``observation.state`` / ``action`` survive.

    Reads ``info.json`` once, finds the ``names`` list for each feature in
    ``STATE_ACTION_KEYS``, and returns ``(keep_indices, original_dims)``
    where ``keep_indices[k]`` is the list of indices to retain (sorted,
    ascending) and ``original_dims[k]`` is the pre-slicing dimensionality.

    With ``exclude_prefixes`` empty this returns ``({}, {})`` so callers can
    treat "no arm filter" as a strict no-op.

    Refuses (``SystemExit``) to:
        - leave a feature with zero dimensions, or
        - end up with mismatched joint sets between ``observation.state`` and
          ``action`` (which would silently desynchronize state vs action).
    """
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"Cannot derive arm indices: missing {info_path}")

    info = json.loads(info_path.read_text())
    features = info.get("features") or {}

    if not exclude_prefixes:
        return {}, {}

    keep: dict[str, list[int]] = {}
    orig_dims: dict[str, int] = {}
    kept_name_lists: dict[str, list[str]] = {}
    for feature_key in STATE_ACTION_KEYS:
        ft = features.get(feature_key)
        if not isinstance(ft, dict):
            continue
        names = list(ft.get("names") or [])
        if not names:
            raise SystemExit(
                f"--exclude-arms requires `names` on feature '{feature_key}' "
                f"in {info_path}; got none. Cannot resolve arm membership."
            )
        kept_indices = [
            i for i, n in enumerate(names)
            if not _arm_prefix_matches(str(n), exclude_prefixes)
        ]
        if not kept_indices:
            raise SystemExit(
                f"--exclude-arms {sorted(exclude_prefixes)} drops every joint "
                f"of '{feature_key}'. Refusing to write a 0-D feature."
            )
        keep[feature_key] = kept_indices
        orig_dims[feature_key] = len(names)
        kept_name_lists[feature_key] = [str(names[i]) for i in kept_indices]

    if "observation.state" in kept_name_lists and "action" in kept_name_lists:
        if kept_name_lists["observation.state"] != kept_name_lists["action"]:
            raise SystemExit(
                "--exclude-arms produced inconsistent slices for "
                "observation.state vs action. "
                f"state-kept={kept_name_lists['observation.state']}, "
                f"action-kept={kept_name_lists['action']}."
            )

    return keep, orig_dims


def _slice_table_state_action(
    table: pa.Table,
    keep: dict[str, list[int]],
) -> pa.Table:
    """Return ``table`` with state/action list columns sliced to ``keep`` indices.

    Columns absent from the table or absent from ``keep`` are left
    untouched. The element type of each column is preserved.
    """
    if not keep:
        return table
    for col_name, idx in keep.items():
        if col_name not in table.schema.names:
            continue
        field = table.schema.field(col_name)
        elem_type = field.type.value_type
        rows = table.column(col_name).to_pylist()
        sliced_values = [
            None if row is None else [row[i] for i in idx]
            for row in rows
        ]
        sliced = pa.array(sliced_values, type=pa.list_(elem_type))
        table = table.set_column(
            table.schema.get_field_index(col_name),
            col_name,
            sliced,
        )
    return table


def rename_in_string(s: str) -> str:
    for old, new in KEY_MAP.items():
        if old in s:
            s = s.replace(old, new)
    return s


def renamed_schema(schema: pa.Schema) -> pa.Schema:
    fields = [
        pa.field(
            rename_in_string(field.name),
            field.type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        for field in schema
    ]
    return pa.schema(fields, metadata=schema.metadata)


def trim_data_index(data_index: pd.DataFrame, skip_first_frames: int) -> pd.DataFrame:
    if skip_first_frames <= 0:
        trimmed = data_index.copy()
    else:
        episode_positions = data_index.groupby("episode_index", sort=False).cumcount()
        trimmed = data_index.loc[episode_positions >= skip_first_frames].copy()

    if trimmed.empty:
        raise SystemExit("Skipping that many frames would remove the entire dataset")

    missing_episodes = set(data_index["episode_index"].unique()) - set(trimmed["episode_index"].unique())
    if missing_episodes:
        episodes = ", ".join(str(ep) for ep in sorted(int(ep) for ep in missing_episodes))
        raise SystemExit(
            f"Skipping {skip_first_frames} frame(s) would empty episode(s): {episodes}"
        )

    trimmed["frame_index"] = trimmed.groupby("episode_index", sort=False).cumcount()
    trimmed["index"] = range(len(trimmed))
    return trimmed


def delete_episodes(data_index: pd.DataFrame, episode_ids: set[int]) -> pd.DataFrame:
    if not episode_ids:
        return data_index.copy()

    available = {int(episode_id) for episode_id in data_index["episode_index"].unique()}
    missing = episode_ids - available
    if missing:
        episodes = ", ".join(str(ep) for ep in sorted(missing))
        raise SystemExit(f"Requested episode(s) do not exist in the source dataset: {episodes}")

    filtered = data_index.loc[~data_index["episode_index"].isin(episode_ids)].copy()
    if filtered.empty:
        raise SystemExit("Deleting those episode(s) would remove the entire dataset")
    return filtered


def remap_episode_indices(data_index: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    episode_ids = sorted(int(episode_id) for episode_id in data_index["episode_index"].unique())
    mapping = {episode_id: new_episode_id for new_episode_id, episode_id in enumerate(episode_ids)}
    remapped = data_index.copy()
    remapped["episode_index"] = remapped["episode_index"].map(mapping)
    return remapped, mapping


def summarize_trimmed_data(trimmed: pd.DataFrame) -> dict[int, dict[str, int]]:
    summary: dict[int, dict[str, int]] = {}
    for episode_id, group in trimmed.groupby("episode_index", sort=False):
        start_index = int(group["index"].min())
        end_index = int(group["index"].max()) + 1
        summary[int(episode_id)] = {
            "length": int(len(group)),
            "dataset_from_index": start_index,
            "dataset_to_index": end_index,
        }
    return summary


def write_trimmed_data(
    src: Path,
    dst: Path,
    trimmed: pd.DataFrame,
    source_tables: dict[str, pa.Table],
    skip_first_frames: int = 0,
    fps: int | None = None,
) -> None:
    frame_offset = 0.0
    if skip_first_frames > 0:
        if fps is None:
            raise ValueError("fps is required when skip_first_frames is set")
        frame_offset = skip_first_frames / float(fps)

    for rel, table in source_tables.items():
        group = trimmed.loc[trimmed["__source_relpath"] == rel].sort_values("__source_row")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        if group.empty:
            pq.write_table(table.slice(0, 0), out)
            continue

        source_rows = pa.array(group["__source_row"].tolist(), type=pa.int64())
        selected = table.take(source_rows)
        if "episode_index" in selected.schema.names:
            selected = selected.set_column(
                selected.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array(
                    group["episode_index"].tolist(),
                    type=selected.schema.field("episode_index").type,
                ),
            )
        if skip_first_frames > 0 and "timestamp" in selected.schema.names:
            timestamps = selected.column("timestamp").to_pylist()
            selected = selected.set_column(
                selected.schema.get_field_index("timestamp"),
                "timestamp",
                pa.array(
                    [float(ts) - frame_offset for ts in timestamps],
                    type=selected.schema.field("timestamp").type,
                ),
            )
        frame_index_field = selected.schema.get_field_index("frame_index")
        index_field = selected.schema.get_field_index("index")
        selected = selected.set_column(
            frame_index_field,
            "frame_index",
            pa.array(group["frame_index"].tolist(), type=selected.schema.field("frame_index").type),
        )
        selected = selected.set_column(
            index_field,
            "index",
            pa.array(group["index"].tolist(), type=selected.schema.field("index").type),
        )
        if ARM_KEEP_INDICES:
            selected = _slice_table_state_action(selected, ARM_KEEP_INDICES)
        pq.write_table(selected, out)


def rewrite_info_json(
    src: Path,
    dst: Path,
    downsample: int,
    total_frames: int | None = None,
    total_episodes: int | None = None,
) -> None:
    info = read_info(src)
    new_features = {}
    for key, ft in info["features"].items():
        if ft.get("dtype") == "video":
            # Drop video features that were excluded from KEY_MAP (e.g. via
            # --exclude-cameras, or unprefixed keys we chose not to carry).
            if _CAMERA_PREFIX_RE.match(key) is not None and key not in KEY_MAP:
                continue
            new_key = KEY_MAP.get(key, key)
            ft = json.loads(json.dumps(ft))  # deep copy
            h, w, c = ft["shape"]
            if OUT_SIZE is not None:
                new_h, new_w = OUT_SIZE
            else:
                new_h, new_w = h // downsample, w // downsample
            ft["shape"] = [new_h, new_w, c]
            if "info" in ft:
                ft["info"]["video.height"] = new_h
                ft["info"]["video.width"] = new_w
                ft["info"]["video.codec"] = "h264"
                ft["info"]["video.pix_fmt"] = "yuv420p"
                # Record the source ROI so deploy/dagger can reproduce the same
                # full-res crop-then-resize. Uncropped cameras get the full
                # source frame so downstream code can treat every camera
                # uniformly.
                cam_name = _stripped_cam_name(new_key)
                spec = CROPS.get(cam_name) or {
                    "top": 0,
                    "left": 0,
                    "height": int(h),
                    "width": int(w),
                }
                ft["info"]["source_crop"] = dict(spec)
                ft["info"]["source_shape"] = [int(h), int(w)]
        else:
            new_key = KEY_MAP.get(key, key)
            if key in ARM_KEEP_INDICES:
                # Shrink shape[0] and filter `names` to the kept arm joints.
                # Other shape dims (typically none for state/action) pass
                # through untouched so this stays generic if the schema
                # later grows extra leading axes.
                keep = ARM_KEEP_INDICES[key]
                ft = json.loads(json.dumps(ft))
                shape = list(ft.get("shape") or [])
                if shape and isinstance(shape[0], int):
                    ft["shape"] = [len(keep)] + shape[1:]
                names = ft.get("names")
                if isinstance(names, list):
                    ft["names"] = [names[i] for i in keep]
        new_features[new_key] = ft
    info["features"] = new_features
    if total_frames is not None:
        info["total_frames"] = total_frames
    if total_episodes is not None:
        info["total_episodes"] = total_episodes
        splits = info.get("splits")
        if isinstance(splits, dict):
            new_splits = {}
            for split_name, split_range in splits.items():
                if isinstance(split_range, str) and ":" in split_range:
                    start_str, end_str = split_range.split(":", 1)
                    if start_str.isdigit() and end_str.isdigit() and int(end_str) > total_episodes:
                        new_splits[split_name] = f"{start_str}:{total_episodes}"
                    else:
                        new_splits[split_name] = split_range
                else:
                    new_splits[split_name] = split_range
            info["splits"] = new_splits
    write_info(dst, info)


def rewrite_stats_json(src: Path, dst: Path, total_frames: int | None = None) -> None:
    stats = read_stats(src)
    if stats is None:
        return
    new_stats: dict[str, object] = {}
    for k, v in stats.items():
        # Drop stats for excluded camera_NN_ keys; pass non-video / already-stripped keys through.
        if _CAMERA_PREFIX_RE.match(k) is not None and k not in KEY_MAP:
            continue
        new_stats[KEY_MAP.get(k, k)] = v
    if total_frames is not None:
        for key, value in list(new_stats.items()):
            if key.endswith("/count"):
                new_stats[key] = replace_count_leaves(value, total_frames)

    # Slice per-axis stats for the affected state/action features. Each
    # entry is a dict like ``{"min": [...], "max": [...], ...}`` with one
    # value per dimension; we keep only the indices that survived the
    # arm filter. Non-list / wrong-length values are passed through so we
    # don't corrupt scalar entries (e.g. legacy stats with a single count).
    if ARM_KEEP_INDICES:
        for feature_key, keep in ARM_KEEP_INDICES.items():
            entry = new_stats.get(feature_key)
            if not isinstance(entry, dict):
                continue
            orig_dim = ARM_ORIG_DIMS.get(feature_key)
            sliced_entry: dict[str, object] = {}
            for stat_name, value in entry.items():
                if (
                    isinstance(value, list)
                    and orig_dim is not None
                    and len(value) == orig_dim
                ):
                    sliced_entry[stat_name] = [value[i] for i in keep]
                else:
                    sliced_entry[stat_name] = value
            new_stats[feature_key] = sliced_entry

    write_stats(dst, new_stats)


def _is_excluded_video_column(column_name: str) -> bool:
    """True if a parquet column belongs to a camera_NN_ feature we are not keeping.

    ``episodes.parquet`` stores per-episode video metadata columns named like
    ``videos/observation.images.camera_NN_<name>/from_timestamp``. Columns
    referencing any ``camera_NN_<name>`` whose feature key is *not* in
    ``KEY_MAP`` (i.e. excluded via ``--exclude-cameras``) should be dropped.
    """
    if "observation.images.camera_" not in column_name:
        return False
    match = re.search(r"observation\.images\.camera_\d+_[^/]+", column_name)
    if match is None:
        return False
    feature_key = match.group(0)
    return feature_key not in KEY_MAP


def rewrite_episodes_parquet(
    src: Path,
    dst: Path,
    episode_summary: dict[int, dict[str, int]] | None = None,
    episode_id_map: dict[int, int] | None = None,
    deleted_episodes: set[int] | None = None,
    skip_first_frames: int = 0,
    fps: int | None = None,
) -> None:
    src_dir = src / "meta" / "episodes"
    if not src_dir.exists():
        return
    for p in src_dir.rglob("*.parquet"):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(p)

        # Drop columns for excluded cameras *before* renaming, while the
        # camera_NN_ prefixes are still present (it's how we identify them).
        keep_cols = [name for name in table.schema.names if not _is_excluded_video_column(name)]
        if len(keep_cols) != len(table.schema.names):
            table = table.select(keep_cols)

        frame = table.to_pandas()
        frame = frame.rename(columns=rename_in_string)

        # Slice per-episode stats columns for state/action down to the
        # surviving arm indices. Columns are list-typed (variable length)
        # in the source schema, so we don't need to rebuild the schema —
        # rewriting shorter lists into the same field type is allowed.
        if ARM_KEEP_INDICES:
            for col in list(frame.columns):
                match = _EPISODE_STATE_ACTION_COL_RE.match(col)
                if match is None:
                    continue
                feature_key = match.group("feature")
                keep = ARM_KEEP_INDICES.get(feature_key)
                orig_dim = ARM_ORIG_DIMS.get(feature_key)
                if keep is None or orig_dim is None:
                    continue

                def _slice_row(value, keep=keep, orig_dim=orig_dim):
                    if value is None:
                        return value
                    seq = list(value)
                    if len(seq) != orig_dim:
                        # Unexpected per-row length — leave it alone so we
                        # don't silently corrupt the file.
                        return seq
                    return [seq[i] for i in keep]

                frame[col] = frame[col].apply(_slice_row)

        if deleted_episodes and "episode_index" in frame.columns:
            frame = frame.loc[~frame["episode_index"].isin(deleted_episodes)].copy()

        if episode_id_map and "episode_index" in frame.columns:
            frame["episode_index"] = frame["episode_index"].map(episode_id_map)

        if frame.empty:
            pq.write_table(table.slice(0, 0), out)
            continue

        if episode_summary is not None:
            frame["length"] = frame["episode_index"].map(
                lambda episode_id: episode_summary[int(episode_id)]["length"]
            )
            frame["dataset_from_index"] = frame["episode_index"].map(
                lambda episode_id: episode_summary[int(episode_id)]["dataset_from_index"]
            )
            frame["dataset_to_index"] = frame["episode_index"].map(
                lambda episode_id: episode_summary[int(episode_id)]["dataset_to_index"]
            )

            if skip_first_frames > 0:
                if fps is None:
                    raise ValueError("fps is required when skip_first_frames is set")
                frame_offset = skip_first_frames / float(fps)
                for new_key in KEY_MAP.values():
                    from_col = f"videos/{new_key}/from_timestamp"
                    to_col = f"videos/{new_key}/to_timestamp"
                    if from_col not in frame.columns or to_col not in frame.columns:
                        continue
                    frame[from_col] = frame[from_col] + frame_offset
                    frame[to_col] = frame[from_col] + frame["length"] / float(fps)

        renamed = renamed_schema(table.schema)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False, schema=renamed), out)


def copy_data_parquets(src: Path, dst: Path) -> None:
    """Mirror ``data/`` parquets to the destination, slicing arms if requested.

    With no ``--exclude-arms`` filter active, this is the original ``copy2``
    fast path. Otherwise we read each parquet, slice the
    ``observation.state`` / ``action`` columns, and rewrite — the data layout
    is otherwise unchanged.
    """
    src_dir = src / "data"
    if not src_dir.exists():
        return
    for p in src_dir.rglob("*.parquet"):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if ARM_KEEP_INDICES:
            table = pq.read_table(p)
            table = _slice_table_state_action(table, ARM_KEEP_INDICES)
            pq.write_table(table, out)
        else:
            shutil.copy2(p, out)


def copy_tasks(src: Path, dst: Path) -> None:
    for name in ("tasks.parquet", "subtasks.parquet"):
        p = src / "meta" / name
        if p.exists():
            shutil.copy2(p, dst / "meta" / name)


def downsample_videos(src: Path, dst: Path, downsample: int) -> None:
    src_videos = src / "videos"
    if not src_videos.exists():
        return
    for old_key, new_key in KEY_MAP.items():
        src_key_dir = src_videos / old_key
        if not src_key_dir.exists():
            print(f"[skip] {src_key_dir} not present")
            continue
        cam_name = _stripped_cam_name(new_key)
        crop_spec = CROPS.get(cam_name)
        for mp4 in src_key_dir.rglob("*.mp4"):
            rel = mp4.relative_to(src_key_dir)
            out = dst / "videos" / new_key / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if OUT_SIZE is not None:
                out_h, out_w = OUT_SIZE
                # crop=w:h:x:y takes x=left, y=top. Apply the full-res crop
                # first, then scale to the common output size (even dims).
                scale = f"scale=trunc({out_w}/2)*2:trunc({out_h}/2)*2"
                if crop_spec is not None:
                    vf = (
                        f"crop={crop_spec['width']}:{crop_spec['height']}:"
                        f"{crop_spec['left']}:{crop_spec['top']},{scale}"
                    )
                else:
                    vf = scale
            else:
                # iw/downsample, ih/downsample, force even dims
                vf = (
                    f"scale=trunc(iw/{downsample}/2)*2:trunc(ih/{downsample}/2)*2"
                )
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-an",
                str(out),
            ]
            print(f"[ffmpeg] {mp4} -> {out}")
            subprocess.run(cmd, check=True)


def write_experiment_config(src: Path, dst: Path) -> None:
    """Copy ``experiment_config.yaml``, dropping any excluded arm block.

    With no ``--exclude-arms`` filter active this is a straight ``copy2``.
    Otherwise we parse the YAML, drop any top-level mapping whose key
    matches an excluded prefix (e.g. ``left_arm`` for ``--exclude-arms left``),
    and write the filtered version. Keeping the destination's
    ``experiment_config.yaml`` consistent with the new state/action
    dimensionality means downstream tools (``record.py``, ``deploy.py``,
    ``app.py``) won't reference a phantom arm.
    """
    p = src / "experiment_config.yaml"
    if not p.exists():
        return
    out = dst / "experiment_config.yaml"
    if not EXCLUDED_ARM_PREFIXES:
        shutil.copy2(p, out)
        return

    text = p.read_text()
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        print(
            f"[warn] could not parse {p} ({exc}); copying unchanged so "
            "experiment_config is at least preserved."
        )
        shutil.copy2(p, out)
        return

    if not isinstance(cfg, dict):
        shutil.copy2(p, out)
        return

    removed: list[str] = []
    for key in list(cfg.keys()):
        if isinstance(key, str) and _arm_prefix_matches(key, EXCLUDED_ARM_PREFIXES):
            removed.append(key)
            cfg.pop(key)
    if removed:
        print(f"[info] experiment_config.yaml: removed top-level key(s) {removed}")

    out.write_text(yaml.safe_dump(cfg, sort_keys=False))


def copy_extras(src: Path, dst: Path) -> None:
    for name in ("recording_config.yaml", "README.md"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    write_experiment_config(src, dst)


def clean_destination(dst: Path) -> None:
    """Remove generated outputs so reruns don't accumulate stale files."""
    managed_dirs = [
        dst / "data",
        dst / "videos",
        dst / "meta" / "episodes",
    ]
    for directory in managed_dirs:
        if directory.exists():
            shutil.rmtree(directory)

    managed_files = [
        dst / "meta" / "info.json",
        dst / "meta" / "stats.json",
        dst / "meta" / "tasks.parquet",
        dst / "meta" / "subtasks.parquet",
    ]
    for file_path in managed_files:
        if file_path.exists():
            file_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Source LeRobot v3.0 dataset root")
    ap.add_argument("--dst", type=Path, required=True, help="Destination dataset root")
    ap.add_argument("--downsample", type=int, default=5, help="Spatial downsample factor (default 5)")
    ap.add_argument(
        "--out-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help=(
            "Common output resolution H W that every camera is scaled to after "
            "any --crop. When set, this overrides the 1/downsample sizing so all "
            "cameras share one shape (required by the diffusion policy)."
        ),
    )
    ap.add_argument(
        "--crop",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "TOP", "LEFT", "HEIGHT", "WIDTH"),
        help=(
            "Full-res crop ROI for one camera, e.g. "
            "--crop right_wrist_top 3 436 704 505. NAME is the stripped camera "
            "name (after the camera_NN_ prefix). Repeatable. Cameras without a "
            "--crop are scaled full-frame. The ROI is recorded in the output "
            "info.json so deploy can reproduce it."
        ),
    )
    ap.add_argument(
        "--skip-first-frames",
        "--discard-first-frames",
        dest="skip_first_frames",
        type=int,
        default=0,
        help="Discard this many frames from the start of every episode before rewriting",
    )
    ap.add_argument(
        "--delete-episodes",
        nargs="+",
        type=int,
        default=[],
        help="Remove these episode indices from the rewritten dataset",
    )
    ap.add_argument(
        "--exclude-cameras",
        nargs="+",
        default=[],
        metavar="NAME",
        help=(
            "Stripped camera names to drop entirely from the destination "
            "(e.g. --exclude-cameras back right_wrist_top). Match the name "
            "after the camera_NN_ prefix has been stripped."
        ),
    )
    ap.add_argument(
        "--exclude-arms",
        nargs="+",
        default=[],
        metavar="PREFIX",
        help=(
            "Joint-name prefixes whose dimensions are removed from "
            "observation.state and action (e.g. --exclude-arms left to keep "
            "only the right arm). Matched case-insensitively against the "
            "`names` listed in info.json; trailing underscores on the prefix "
            "are ignored. The matching top-level block in "
            "experiment_config.yaml is also stripped from the destination."
        ),
    )
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    require_dataset(src)
    if src == dst:
        raise SystemExit("--src and --dst must be different paths")
    dst.mkdir(parents=True, exist_ok=True)
    clean_destination(dst)
    source_info = read_info(src)
    source_fps = int(source_info["fps"])

    # Auto-derive the camera rename map from the source dataset, dropping any
    # cameras the user excluded on the CLI. KEY_MAP and EXCLUDED_CAMERAS are
    # module-level so the helper functions above pick them up.
    global KEY_MAP, EXCLUDED_CAMERAS, EXCLUDED_ARM_PREFIXES, ARM_KEEP_INDICES, ARM_ORIG_DIMS
    global CROPS, OUT_SIZE
    EXCLUDED_CAMERAS = {str(name).strip() for name in args.exclude_cameras if str(name).strip()}
    KEY_MAP = derive_key_map(src, exclude=EXCLUDED_CAMERAS)

    CROPS = parse_crops(args.crop)
    OUT_SIZE = (int(args.out_size[0]), int(args.out_size[1])) if args.out_size else None
    if CROPS and OUT_SIZE is None:
        raise SystemExit("--crop requires --out-size so all cameras share one shape")
    if CROPS:
        validate_crops_against_source(source_info, KEY_MAP, CROPS)
        print("[info] per-camera full-res crops:")
        for name, spec in CROPS.items():
            print(f"         {name}: {spec}")
    if OUT_SIZE is not None:
        print(f"[info] common output size (HxW): {OUT_SIZE[0]}x{OUT_SIZE[1]}")

    EXCLUDED_ARM_PREFIXES = {
        str(p).strip() for p in args.exclude_arms if str(p).strip()
    }
    ARM_KEEP_INDICES, ARM_ORIG_DIMS = derive_arm_indices(src, EXCLUDED_ARM_PREFIXES)

    if EXCLUDED_ARM_PREFIXES:
        src_features = source_info.get("features") or {}
        print(f"[info] --exclude-arms = {sorted(EXCLUDED_ARM_PREFIXES)}")
        for feature_key in STATE_ACTION_KEYS:
            keep = ARM_KEEP_INDICES.get(feature_key)
            ft = src_features.get(feature_key) or {}
            names = ft.get("names") or []
            if keep is None or not names:
                continue
            kept_set = set(keep)
            dropped = [str(n) for i, n in enumerate(names) if i not in kept_set]
            print(
                f"[info] {feature_key}: dropping {len(dropped)} joint(s) "
                f"({', '.join(dropped)}); shape {len(names)} -> {len(keep)}"
            )

    print(
        f"[info] src={src}\n[info] dst={dst}\n[info] downsample={args.downsample}x\n"
        f"[info] skip_first_frames={args.skip_first_frames}"
    )

    if args.skip_first_frames < 0:
        raise SystemExit("--skip-first-frames must be >= 0")

    delete_episode_ids = {int(episode_id) for episode_id in args.delete_episodes}

    episode_summary: dict[int, dict[str, int]] | None = None
    total_frames: int | None = None
    episode_id_map: dict[int, int] | None = None
    total_episodes: int | None = None

    if args.skip_first_frames > 0 or delete_episode_ids:
        data_index, source_tables = load_data_index(
            src,
            rel_col="__source_relpath",
            row_col="__source_row",
            allow_empty=True,
        )
        data_index = delete_episodes(data_index, delete_episode_ids)
        if delete_episode_ids:
            data_index, episode_id_map = remap_episode_indices(data_index)
        trimmed = trim_data_index(data_index, args.skip_first_frames)
        episode_summary = summarize_trimmed_data(trimmed)
        total_frames = len(trimmed)
        total_episodes = len(episode_summary)
        write_trimmed_data(
            src,
            dst,
            trimmed,
            source_tables,
            skip_first_frames=args.skip_first_frames,
            fps=source_fps,
        )
    else:
        copy_data_parquets(src, dst)

    rewrite_info_json(
        src,
        dst,
        args.downsample,
        total_frames=total_frames,
        total_episodes=total_episodes,
    )
    rewrite_stats_json(src, dst, total_frames=total_frames)
    copy_tasks(src, dst)
    rewrite_episodes_parquet(
        src,
        dst,
        episode_summary=episode_summary,
        episode_id_map=episode_id_map,
        deleted_episodes=delete_episode_ids,
        skip_first_frames=args.skip_first_frames,
        fps=source_fps,
    )
    downsample_videos(src, dst, args.downsample)
    copy_extras(src, dst)
    print("[done]")


if __name__ == "__main__":
    main()
