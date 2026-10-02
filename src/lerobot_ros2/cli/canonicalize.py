#!/usr/bin/env python
"""Canonicalize a LeRobot v3.0 dataset to the bimanual 3-camera LBM schema.

The short-horizon datasets in this project were recorded under three
different schemas: bimanual (14-dim, ``front`` + both wrist cameras),
left-arm-only (7-dim, ``front`` + ``left_wrist_top``) and right-arm-only
(7-dim, ``front`` + ``right_wrist_top``). They cannot be merged into a single
pretraining pool until every one of them exposes the *same* features, so this
script rewrites a dataset in place-equivalent form with:

* ``observation.state`` / ``action`` widened to 14 dims
  (``left_*`` joints then ``right_*`` joints). The arm that was not recorded is
  filled with a constant **park pose** rather than zeros -- an all-zero joint
  vector is a real, reachable configuration, so padding with zeros would teach
  the policy to drive the idle arm into it. The defaults are poses actually
  observed being held by the idle arm in the long-horizon recordings.
* all three cameras present. A camera that was never recorded gets a
  synthesized all-black video track whose per-file frame counts mirror an
  existing camera, so the per-episode video timestamps carry over unchanged.
  Black is deliberate: it marks the modality as absent instead of inventing
  plausible-but-wrong pixels.
* an ``action_source`` column on every dataset (``1`` == human teleop), so the
  pool can be trained with ``lerobot-ros-train-dagger``, which keeps
  ``action_source == 0`` frames as observation context but never samples them
  as loss anchors. Datasets that already carry plateau tags keep them.
* a single task string, and h264 video.

Videos are copied byte-for-byte unless a re-encode is requested, and episode
dropping only rewrites parquet rows -- the packed video files keep their
unused footage, which stays harmless because episodes address video by
timestamp.

Examples::

    lerobot-ros-canonicalize \
        --src /lerobot-ros/data/smrithi/stir_bar/policy/train/stir_bar_20260505_ds5 \
        --dst /lerobot-ros/data/rama/lbm_canon/stir_bar \
        --task-name stir_bar

    # right-arm dataset that also carries DAgger episodes and AV1 video
    lerobot-ros-canonicalize \
        --src .../septum.../train --dst .../lbm_canon/septum \
        --task-name septum_insert_reactor \
        --drop-dagger-episodes --reencode-video
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_ros2.dataset_rewrite import (
    load_data_index,
    read_info,
    read_stats,
    require_dataset,
    write_info,
    write_stats,
)

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "robotiq_85_left_knuckle_joint",
)
CANON_JOINT_NAMES = [f"left_{j}" for j in ARM_JOINTS] + [f"right_{j}" for j in ARM_JOINTS]
CANON_CAMERAS = ("front", "left_wrist_top", "right_wrist_top")
STATE_ACTION_KEYS = ("observation.state", "action")
IMG_PREFIX = "observation.images."

# Poses actually observed being held by the idle arm during the long-horizon
# recordings (selected as the real hold vector closest to the length-weighted
# median hold, so they are guaranteed reachable and collision-free).
DEFAULT_PARK_LEFT = [2.255082, -1.578954, 1.338951, -1.296062, -1.514087, 0.710721, 0.0]
DEFAULT_PARK_RIGHT = [0.100741, -2.157641, -1.363733, -1.230887, 1.560952, 0.075859, 0.332335]

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# ── source inspection ────────────────────────────────────────────────────


def detect_layout(info: dict) -> dict:
    """Work out what has to change to bring ``info`` up to the canonical schema."""
    features = info["features"]
    names = list(features["action"].get("names") or [])
    dim = int(features["action"]["shape"][0])
    if len(names) != dim:
        raise SystemExit(f"action has shape {dim} but {len(names)} names; cannot resolve arms")

    if dim == 14:
        if names != CANON_JOINT_NAMES:
            raise SystemExit(f"14-dim dataset has unexpected joint order:\n  {names}")
        side = None
    elif dim == 7:
        if all(n.startswith("left_") for n in names):
            side = "left"
        elif all(n.startswith("right_") for n in names):
            side = "right"
        else:
            raise SystemExit(f"7-dim dataset mixes arms: {names}")
        if names != [f"{side}_{j}" for j in ARM_JOINTS]:
            raise SystemExit(f"7-dim dataset has unexpected joint order:\n  {names}")
    else:
        raise SystemExit(f"unsupported action dim {dim}")

    cams = sorted(k[len(IMG_PREFIX):] for k in features if k.startswith(IMG_PREFIX))
    unknown = set(cams) - set(CANON_CAMERAS)
    if unknown:
        raise SystemExit(f"unknown camera(s) {sorted(unknown)}; expected {CANON_CAMERAS}")
    missing = [c for c in CANON_CAMERAS if c not in cams]
    if len(missing) > 1:
        raise SystemExit(f"more than one camera missing ({missing}); refusing to synthesize")

    return {
        "side": side,
        "dim": dim,
        "cameras": cams,
        "missing_camera": missing[0] if missing else None,
        "has_action_source": "action_source" in features,
    }


def pad_indices(side: str | None) -> tuple[list[int], list[int]]:
    """``(target_slots, park_slots)`` for the recorded arm and the filled arm."""
    if side is None:
        return list(range(14)), []
    if side == "left":
        return list(range(0, 7)), list(range(7, 14))
    return list(range(7, 14)), list(range(0, 7))


def park_vector(side: str | None, park_left: list[float], park_right: list[float]) -> list[float]:
    """The full 14-dim vector with both arms parked (recorded dims overwritten later)."""
    return list(park_left) + list(park_right)


# ── dagger detection ─────────────────────────────────────────────────────


def dagger_episodes(df: pd.DataFrame) -> set[int]:
    """Episodes whose ``action_source == 0`` frames form a contiguous prefix.

    That is the signature ``dagger.py`` leaves behind: it flushes a fixed-size
    pre-intervention buffer of policy frames at the start of the episode and
    then records human frames. Plateau tagging, which also writes zeros, marks
    *low-motion* frames scattered through the episode, so it does not match.
    """
    if "action_source" not in df.columns:
        return set()
    out: set[int] = set()
    for ep, grp in df.groupby("episode_index", sort=True):
        v = grp.sort_values("frame_index")["action_source"].to_numpy().astype(np.int64)
        if not (v == 0).any() or (v == 0).all():
            continue
        first_human = int(np.argmax(v == 1))
        if first_human > 0 and (v[:first_human] == 0).all() and (v[first_human:] == 1).all():
            out.add(int(ep))
    return out


# ── data parquet rewrite ─────────────────────────────────────────────────


def widen_table(
    table: pa.Table,
    side: str | None,
    park_left: list[float],
    park_right: list[float],
    action_source_value: int | None,
) -> pa.Table:
    """Widen state/action to fixed-size 14 and add ``action_source`` if absent."""
    keep, _ = pad_indices(side)
    base = park_vector(side, park_left, park_right)

    for col in STATE_ACTION_KEYS:
        if col not in table.schema.names:
            continue
        rows = table.column(col).to_pylist()
        widened = []
        for row in rows:
            if row is None:
                widened.append(None)
                continue
            if len(row) == 14:
                widened.append([float(x) for x in row])
                continue
            if len(row) != len(keep):
                raise SystemExit(f"{col}: expected {len(keep)} dims, got {len(row)}")
            vec = list(base)
            for slot, value in zip(keep, row):
                vec[slot] = float(value)
            widened.append(vec)
        arr = pa.array(widened, type=pa.list_(pa.float32(), 14))
        table = table.set_column(table.schema.get_field_index(col), col, arr)

    if action_source_value is not None and "action_source" not in table.schema.names:
        col = pa.array([action_source_value] * table.num_rows, type=pa.int64())
        table = table.append_column("action_source", col)

    return table


def hf_metadata_for(info: dict) -> bytes:
    """Rebuild the ``huggingface`` parquet schema metadata from ``info.json``."""
    feats: dict = {}
    for key, ft in info["features"].items():
        if ft.get("dtype") == "video":
            continue
        shape = list(ft.get("shape") or [])
        if ft["dtype"] in ("float32", "float64") and shape and shape[0] > 1:
            feats[key] = {
                "feature": {"dtype": ft["dtype"], "_type": "Value"},
                "length": int(shape[0]),
                "_type": "List",
            }
        else:
            feats[key] = {"dtype": ft["dtype"], "_type": "Value"}
    return json.dumps({"info": {"features": feats}}).encode()


def write_data(
    src: Path,
    dst: Path,
    kept: pd.DataFrame,
    tables: dict[str, pa.Table],
    side: str | None,
    park_left: list[float],
    park_right: list[float],
    action_source_value: int | None,
    task_index: int,
    hf_meta: bytes,
) -> None:
    for rel, table in tables.items():
        grp = kept.loc[kept["__rel"] == rel].sort_values("__row")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        table = widen_table(table, side, park_left, park_right, action_source_value)
        if grp.empty:
            pq.write_table(table.slice(0, 0).replace_schema_metadata({b"huggingface": hf_meta}), out)
            continue

        selected = table.take(pa.array(grp["__row"].tolist(), type=pa.int64()))
        for col, values in (
            ("episode_index", grp["episode_index"].tolist()),
            ("index", grp["index"].tolist()),
            ("frame_index", grp["frame_index"].tolist()),
        ):
            selected = selected.set_column(
                selected.schema.get_field_index(col),
                col,
                pa.array(values, type=selected.schema.field(col).type),
            )
        # Collapse the source's task vocabulary onto one canonical task.
        selected = selected.set_column(
            selected.schema.get_field_index("task_index"),
            "task_index",
            pa.array([task_index] * selected.num_rows, type=selected.schema.field("task_index").type),
        )
        selected = selected.replace_schema_metadata({b"huggingface": hf_meta})
        pq.write_table(selected, out)


# ── meta rewrite ─────────────────────────────────────────────────────────


def rewrite_info(
    info: dict,
    layout: dict,
    total_frames: int,
    total_episodes: int,
    mirror_camera: str,
    source_shape: tuple[int, int],
) -> dict:
    info = json.loads(json.dumps(info))
    features = info["features"]

    for key in STATE_ACTION_KEYS:
        if key in features:
            features[key]["shape"] = [14]
            features[key]["names"] = list(CANON_JOINT_NAMES)

    missing = layout["missing_camera"]
    if missing:
        template = json.loads(json.dumps(features[f"{IMG_PREFIX}{mirror_camera}"]))
        features[f"{IMG_PREFIX}{missing}"] = template

    # ``validate_all_metadata`` demands byte-identical feature dicts before it
    # will merge, and only some sources recorded the full-res ROI. Every one of
    # these is a plain 720p -> 256x144 downsample with no crop, so filling in
    # the identity crop unifies the pool without losing deploy information.
    for key, ft in features.items():
        if ft.get("dtype") != "video":
            continue
        ft["info"]["video.codec"] = "h264"
        h, w = source_shape
        ft["info"].setdefault("source_crop", {"top": 0, "left": 0, "height": h, "width": w})
        ft["info"].setdefault("source_shape", [h, w])

    features["action_source"] = {"dtype": "int64", "shape": [1], "names": ["action_source"]}

    ordered = {}
    for key in STATE_ACTION_KEYS:
        if key in features:
            ordered[key] = features[key]
    for cam in CANON_CAMERAS:
        ordered[f"{IMG_PREFIX}{cam}"] = features[f"{IMG_PREFIX}{cam}"]
    for key, ft in features.items():
        if key not in ordered:
            ordered[key] = ft
    info["features"] = ordered

    info["total_frames"] = total_frames
    info["total_episodes"] = total_episodes
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def _zero_like_nested(value):
    """Zero out a stat value while preserving its (possibly ragged) nesting.

    Per-episode image stats are stored as nested object arrays like
    ``[[[0.72]], [[0.57]], [[0.42]]]``, which numpy refuses to coerce to a
    float array, so recurse instead of relying on ``np.zeros_like``.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_zero_like_nested(v) for v in value]
    return 0.0


def _pad_stat_list(value, keep: list[int], base: list[float], fill_zero: bool):
    """Widen a 7-entry per-dimension stat row to 14."""
    if value is None:
        return value
    seq = [float(x) for x in np.asarray(value).reshape(-1)]
    if len(seq) == 14:
        return seq
    if len(seq) != len(keep):
        return seq
    vec = [0.0] * 14 if fill_zero else list(base)
    for slot, v in zip(keep, seq):
        vec[slot] = v
    return vec


def rewrite_stats_json(
    src: Path,
    dst: Path,
    layout: dict,
    park_left: list[float],
    park_right: list[float],
    mirror_camera: str,
) -> None:
    stats = read_stats(src)
    if stats is None:
        return
    keep, _ = pad_indices(layout["side"])
    base = park_vector(layout["side"], park_left, park_right)

    if layout["side"] is not None:
        for key in STATE_ACTION_KEYS:
            entry = stats.get(key)
            if not isinstance(entry, dict):
                continue
            for stat_name, value in list(entry.items()):
                if stat_name == "count":
                    continue
                entry[stat_name] = _pad_stat_list(value, keep, base, fill_zero=(stat_name == "std"))

    missing = layout["missing_camera"]
    if missing:
        mirror = stats.get(f"{IMG_PREFIX}{mirror_camera}", {})
        black: dict = {}
        for stat_name, value in mirror.items():
            black[stat_name] = value if stat_name == "count" else _zero_like_nested(value)
        stats[f"{IMG_PREFIX}{missing}"] = black

    # Match the convention of the datasets that already carry action_source:
    # the column lives in the data parquets only, never in the stats.
    stats.pop("action_source", None)

    write_stats(dst, stats)


def rewrite_episodes(
    src: Path,
    dst: Path,
    layout: dict,
    park_left: list[float],
    park_right: list[float],
    mirror_camera: str,
    dropped: set[int],
    ep_map: dict[int, int],
    summary: dict[int, dict[str, int]],
    task_name: str,
) -> None:
    src_dir = src / "meta" / "episodes"
    keep, _ = pad_indices(layout["side"])
    base = park_vector(layout["side"], park_left, park_right)
    missing = layout["missing_camera"]

    for p in sorted(src_dir.rglob("*.parquet")):
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(p)
        out_chunk = int(re.search(r"chunk-(\d+)", out.as_posix()).group(1))
        out_file = int(re.search(r"file-(\d+)", out.as_posix()).group(1))

        # action_source stats are not part of the convention we standardize on.
        drop_cols = [n for n in table.schema.names if "action_source" in n]
        if drop_cols:
            table = table.drop(drop_cols)

        frame = table.to_pandas()
        if dropped:
            frame = frame.loc[~frame["episode_index"].isin(dropped)].copy()
        if frame.empty:
            pq.write_table(table.slice(0, 0), out)
            continue
        frame["episode_index"] = frame["episode_index"].map(ep_map)

        if layout["side"] is not None:
            for col in list(frame.columns):
                if not col.startswith("stats/"):
                    continue
                feature = col[len("stats/"):].rsplit("/", 1)
                if len(feature) != 2 or feature[0] not in STATE_ACTION_KEYS:
                    continue
                stat_name = feature[1]
                if stat_name == "count":
                    continue
                frame[col] = frame[col].apply(
                    lambda v, s=stat_name: _pad_stat_list(v, keep, base, fill_zero=(s == "std"))
                )

        # Some sources (the previously-merged ones) carry stale pointers here,
        # naming episode-metadata files that were never written. Training does
        # not read these columns, but the merge does, so stamp the file this row
        # is actually landing in.
        frame["meta/episodes/chunk_index"] = out_chunk
        frame["meta/episodes/file_index"] = out_file

        frame["tasks"] = [[task_name]] * len(frame)
        frame["length"] = frame["episode_index"].map(lambda e: summary[int(e)]["length"])
        frame["dataset_from_index"] = frame["episode_index"].map(
            lambda e: summary[int(e)]["dataset_from_index"]
        )
        frame["dataset_to_index"] = frame["episode_index"].map(
            lambda e: summary[int(e)]["dataset_to_index"]
        )

        fields = list(table.schema)
        if missing:
            src_key = f"{IMG_PREFIX}{mirror_camera}"
            new_key = f"{IMG_PREFIX}{missing}"
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                sc, nc = f"videos/{src_key}/{suffix}", f"videos/{new_key}/{suffix}"
                frame[nc] = frame[sc]
                fields.append(pa.field(nc, table.schema.field(sc).type))
            for suffix in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                sc, nc = f"stats/{src_key}/{suffix}", f"stats/{new_key}/{suffix}"
                if sc not in frame.columns:
                    continue
                if suffix == "count":
                    frame[nc] = frame[sc]
                else:
                    frame[nc] = frame[sc].apply(_zero_like_nested)
                fields.append(pa.field(nc, table.schema.field(sc).type))

        schema = pa.schema(fields, metadata=table.schema.metadata)
        frame = frame[[f.name for f in fields]]
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False, schema=schema), out)


def write_tasks(dst: Path, task_name: str) -> int:
    # ``load_tasks`` reads this with a bare ``pd.read_parquet`` and then names
    # the *index* "task", so the task string has to be the index -- as a plain
    # column it would silently resolve every task lookup to the wrong value.
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_name], name="task"))
    tasks.to_parquet(dst / "meta" / "tasks.parquet")
    return 0


# ── video handling ───────────────────────────────────────────────────────


def probe_frames(path: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def copy_or_reencode_videos(src: Path, dst: Path, reencode: bool) -> None:
    for key_dir in sorted((src / "videos").iterdir()):
        for mp4 in sorted(key_dir.rglob("*.mp4")):
            out = dst / mp4.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            if not reencode:
                shutil.copy2(mp4, out)
                continue
            subprocess.run(
                [FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4),
                 # crf 12 keeps the transcode visually transparent; the merge
                 # concatenates video files per key, so every dataset in the
                 # pool has to share one codec even if that costs a re-encode.
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow",
                 "-crf", "12", "-vsync", "0", "-an", str(out)],
                check=True,
            )
            before, after = probe_frames(mp4), probe_frames(out)
            if before != after:
                raise SystemExit(f"re-encode changed frame count for {mp4}: {before} -> {after}")


def synthesize_black_videos(
    src: Path, dst: Path, missing: str, mirror: str, width: int, height: int, fps: int
) -> None:
    """Write an all-black track for ``missing`` mirroring ``mirror``'s file layout."""
    mirror_dir = src / "videos" / f"{IMG_PREFIX}{mirror}"
    for mp4 in sorted(mirror_dir.rglob("*.mp4")):
        n = probe_frames(mp4)
        out = dst / "videos" / f"{IMG_PREFIX}{missing}" / mp4.relative_to(mirror_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}",
             "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-crf", "23", "-an", str(out)],
            check=True,
        )
        made = probe_frames(out)
        if made != n:
            raise SystemExit(f"black track frame count mismatch for {out}: {made} != {n}")


# ── main ─────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--task-name", required=True, help="single task string for every episode")
    p.add_argument("--park-left", type=float, nargs=7, default=DEFAULT_PARK_LEFT)
    p.add_argument("--park-right", type=float, nargs=7, default=DEFAULT_PARK_RIGHT)
    p.add_argument("--drop-dagger-episodes", action="store_true",
                   help="drop episodes whose action_source==0 frames form a contiguous prefix")
    p.add_argument("--expect-dropped", type=int, default=None,
                   help="fail unless exactly this many episodes are dropped")
    p.add_argument("--reencode-video", action="store_true",
                   help="re-encode existing tracks to h264 (needed for AV1 sources)")
    p.add_argument("--source-shape", type=int, nargs=2, default=(720, 1280), metavar=("H", "W"),
                   help="full-res frame size these videos were downsampled from")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src, dst = args.src.resolve(), args.dst.resolve()
    require_dataset(src)
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"destination exists (use --overwrite): {dst}")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    info = read_info(src)
    layout = detect_layout(info)
    mirror = "front" if "front" in layout["cameras"] else layout["cameras"][0]

    print(f"[canon] {src}")
    print(f"[canon]   arms={layout['dim']}d side={layout['side'] or 'bimanual'} "
          f"cameras={layout['cameras']} missing={layout['missing_camera']} "
          f"action_source={'present' if layout['has_action_source'] else 'ADD=1'}")

    index, tables = load_data_index(
        src,
        columns=("episode_index", "index", "frame_index"),
        optional_columns=("action_source",),
    )

    dropped: set[int] = set()
    if args.drop_dagger_episodes:
        dropped = dagger_episodes(index)
        print(f"[canon]   dagger episodes dropped: {len(dropped)}")
    if args.expect_dropped is not None and len(dropped) != args.expect_dropped:
        raise SystemExit(f"expected to drop {args.expect_dropped} episodes, found {len(dropped)}")

    kept = index.loc[~index["episode_index"].isin(dropped)].copy()
    if kept.empty:
        raise SystemExit("every episode was dropped")

    ep_ids = sorted(int(e) for e in kept["episode_index"].unique())
    ep_map = {old: new for new, old in enumerate(ep_ids)}
    kept["episode_index"] = kept["episode_index"].map(ep_map)
    kept = kept.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    kept["frame_index"] = kept.groupby("episode_index", sort=False).cumcount()
    kept["index"] = range(len(kept))

    summary = {
        int(ep): {
            "length": int(len(g)),
            "dataset_from_index": int(g["index"].min()),
            "dataset_to_index": int(g["index"].max()) + 1,
        }
        for ep, g in kept.groupby("episode_index", sort=True)
    }
    total_frames, total_episodes = len(kept), len(summary)
    print(f"[canon]   episodes {len(ep_ids)} (was {index['episode_index'].nunique()}), frames {total_frames}")

    new_info = rewrite_info(
        info, layout, total_frames, total_episodes, mirror, tuple(args.source_shape)
    )
    write_info(dst, new_info)
    hf_meta = hf_metadata_for(new_info)

    task_index = write_tasks(dst, args.task_name)
    write_data(
        src, dst, kept, tables, layout["side"], args.park_left, args.park_right,
        None if layout["has_action_source"] else 1, task_index, hf_meta,
    )
    rewrite_episodes(
        src, dst, layout, args.park_left, args.park_right, mirror,
        dropped, ep_map, summary, args.task_name,
    )
    rewrite_stats_json(src, dst, layout, args.park_left, args.park_right, mirror)

    copy_or_reencode_videos(src, dst, args.reencode_video)
    if layout["missing_camera"]:
        cam_info = new_info["features"][f"{IMG_PREFIX}{mirror}"]["info"]
        synthesize_black_videos(
            src, dst, layout["missing_camera"], mirror,
            int(cam_info["video.width"]), int(cam_info["video.height"]), int(new_info["fps"]),
        )
        print(f"[canon]   synthesized black track for {layout['missing_camera']}")

    for extra in ("experiment_config.yaml", "recording_config.yaml", "plateau_filter_config.yaml"):
        p = src / extra
        if p.exists():
            shutil.copy2(p, dst / extra)

    print(f"[canon] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
